using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using System.IO;

namespace EyeContactCam;

/// <summary>CPU DeepWarp inference: 64x48 eye crop + six landmark maps + target angle.</summary>
public sealed class NeuralGazeCorrector : IDisposable
{
    private const int W = 64, H = 48;
    private readonly InferenceSession _left;
    private readonly InferenceSession _right;

    public NeuralGazeCorrector()
    {
        var dir = Path.Combine(AppContext.BaseDirectory, "Assets", "Neural");
        var options = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
        _left = new InferenceSession(Path.Combine(dir, "gaze_L.onnx"), options);
        _right = new InferenceSession(Path.Combine(dir, "gaze_R.onnx"), options);
        Probe(_left); Probe(_right);
    }

    public double Correct(Mat frame, Rect eye, Point2f[] sixPoints, bool screenLeft, double pupilX, double pupilY,
        double targetX, double targetY, double strength)
    {
        if (sixPoints.Length != 6) return 0;
        // Network training order: R=36,37,38,39,40,41; L=45,44,43,42,47,46.
        var anchorsPx = screenLeft ? sixPoints : new[] { sixPoints[3],sixPoints[2],sixPoints[1],sixPoints[0],sixPoints[5],sixPoints[4] };
        var eyeLen = Math.Abs(anchorsPx[3].X-anchorsPx[0].X);
        if (eyeLen < 10) return 0;
        var cropW = (int)Math.Round(1.5 * eyeLen);
        var cropH = (int)Math.Round(1.125 * eyeLen);
        var cx = (anchorsPx[0].X+anchorsPx[3].X)/2.0; var cy = (anchorsPx[0].Y+anchorsPx[3].Y)/2.0;
        var cropRect = Clamp(new Rect((int)(cx-cropW/2), (int)(cy-cropH*.58), cropW, cropH), frame.Size());
        if (cropRect.Width < 16 || cropRect.Height < 12) return 0;

        using var crop = new Mat(frame, cropRect); using var resized = new Mat();
        Cv2.Resize(crop, resized, new Size(W, H), 0, 0, InterpolationFlags.Area);
        var image = new DenseTensor<float>(new[] { 1, H, W, 3 });
        for (var y=0;y<H;y++) for(var x=0;x<W;x++)
        {
            var p=resized.At<Vec3b>(y,x); image[0,y,x,0]=p.Item0/255f; image[0,y,x,1]=p.Item1/255f; image[0,y,x,2]=p.Item2/255f;
        }

        var maps = new DenseTensor<float>(new[] { 1, H, W, 12 });
        for(var i=0;i<6;i++)
        {
            var ax=(float)((anchorsPx[i].X-cropRect.X)*W/cropRect.Width);
            var ay=(float)((anchorsPx[i].Y-cropRect.Y)*H/cropRect.Height);
            for(var y=0;y<H;y++) for(var x=0;x<W;x++){maps[0,y,x,2*i]=x-ax;maps[0,y,x,2*i+1]=y-ay;}
        }

        var dx=(targetX-pupilX)*eye.Width*strength; var dy=(targetY-pupilY)*eye.Height*strength;
        var scale=.42*eyeLen;
        var vertical=(float)(-1.5*RadiansToDegrees(Math.Asin(Math.Clamp(dy/scale,-.90,.90))));
        var horizontal=(float)(1.5*RadiansToDegrees(Math.Asin(Math.Clamp(dx/scale,-.90,.90))));
        var angles=new DenseTensor<float>(new[]{1,2}); angles[0,0]=vertical; angles[0,1]=horizontal;
        var session=screenLeft?_right:_left; // mirrored preview: screen-left is the subject's right eye.
        using var results=session.Run(new[] {
            NamedOnnxValue.CreateFromTensor("input_img:0",image),
            NamedOnnxValue.CreateFromTensor("input_fp:0",maps),
            NamedOnnxValue.CreateFromTensor("input_ang:0",angles)
        },new[]{"output:0"});
        var output=results.First().AsTensor<float>(); using var generated=new Mat(H,W,MatType.CV_8UC3);
        for(var y=0;y<H;y++) for(var x=0;x<W;x++) generated.Set(y,x,new Vec3b(ToByte(output[0,y,x,0]),ToByte(output[0,y,x,1]),ToByte(output[0,y,x,2])));
        // Fail-safe: never paste a broken/black model result over the user's face.
        var sourceMean=Cv2.Mean(resized);var generatedMean=Cv2.Mean(generated);
        var srcLum=(sourceMean.Val0+sourceMean.Val1+sourceMean.Val2)/3.0;
        var genLum=(generatedMean.Val0+generatedMean.Val1+generatedMean.Val2)/3.0;
        if(genLum<20||genLum>245||Math.Abs(genLum-srcLum)>75)return 0;
        using var full=new Mat();Cv2.Resize(generated,full,cropRect.Size,0,0,InterpolationFlags.Cubic);
        using var mask=new Mat(cropRect.Size,MatType.CV_8UC1,Scalar.Black);
        var polygon=sixPoints.Select(p=>new Point((int)(p.X-cropRect.X),(int)(p.Y-cropRect.Y))).ToArray();
        Cv2.FillPoly(mask,new[]{polygon},Scalar.White);
        var dilate=Math.Max(1,(int)(eyeLen*.08));using(var kernel=Cv2.GetStructuringElement(MorphShapes.Ellipse,new Size(2*dilate+1,2*dilate+1)))Cv2.Dilate(mask,mask,kernel);
        Cv2.GaussianBlur(mask,mask,new Size(0,0),Math.Max(1,eyeLen*.05));
        using var alpha=new Mat();mask.ConvertTo(alpha,MatType.CV_32FC1,1.0/255.0);
        using var src32=new Mat();using var gen32=new Mat();crop.ConvertTo(src32,MatType.CV_32FC3);full.ConvertTo(gen32,MatType.CV_32FC3);
        var channels=new[]{alpha,alpha,alpha};using var alpha3=new Mat();Cv2.Merge(channels,alpha3);
        using var inv=new Mat();Cv2.Subtract(Scalar.All(1),alpha3,inv);using var a=new Mat();using var b=new Mat();Cv2.Multiply(src32,inv,a);Cv2.Multiply(gen32,alpha3,b);Cv2.Add(a,b,a);a.ConvertTo(crop,MatType.CV_8UC3);
        return Math.Sqrt(dx*dx+dy*dy);
    }

    private static void Probe(InferenceSession s)
    {
        var image=new DenseTensor<float>(new[]{1,H,W,3});var maps=new DenseTensor<float>(new[]{1,H,W,12});var angles=new DenseTensor<float>(new[]{1,2});
        using var _=s.Run(new[]{NamedOnnxValue.CreateFromTensor("input_img:0",image),NamedOnnxValue.CreateFromTensor("input_fp:0",maps),NamedOnnxValue.CreateFromTensor("input_ang:0",angles)},new[]{"output:0"});
    }
    private static byte ToByte(float v)=>(byte)Math.Clamp((int)Math.Round(v*255),0,255);
    private static double RadiansToDegrees(double v)=>v*180/Math.PI;
    private static Rect Clamp(Rect r,Size s){var x=Math.Clamp(r.X,0,s.Width-1);var y=Math.Clamp(r.Y,0,s.Height-1);return new Rect(x,y,Math.Max(1,Math.Min(r.Width,s.Width-x)),Math.Max(1,Math.Min(r.Height,s.Height-y)));}
    public void Dispose(){_left.Dispose();_right.Dispose();}
}
