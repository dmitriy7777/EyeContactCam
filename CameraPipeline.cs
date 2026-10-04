using OpenCvSharp;
using System.IO;
using System.Text.Json;

namespace EyeContactCam;

public sealed class CameraPipeline : IDisposable
{
    private VideoCapture? _capture;
    private readonly CascadeClassifier _face;
    private readonly CascadeClassifier _eyes;
    private readonly List<(double X, double Y)>[] _samples = [[], []];
    private readonly double[] _targetX = [.50, .50];
    private readonly double[] _targetY = [.52, .52];
    private readonly double[] _smoothX = [.50, .50];
    private readonly double[] _smoothY = [.52, .52];
    private readonly Rect[] _smoothEyes = [new(), new()];
    private bool _hasSmoothEyes;
    private bool _calibrating;
    private bool _profileLoaded;

    public int LastEyesFound { get; private set; }
    public double LastCorrectionMagnitude { get; private set; }
    public bool HasCalibration => _profileLoaded;
    private static string ProfilePath => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "EyeContactCam", "profile.json");

    public CameraPipeline()
    {
        var assets = Path.Combine(AppContext.BaseDirectory, "Assets");
        _face = new CascadeClassifier(Path.Combine(assets, "haarcascade_frontalface_default.xml"));
        _eyes = new CascadeClassifier(Path.Combine(assets, "haarcascade_eye_tree_eyeglasses.xml"));
        LoadCalibration();
    }

    public void Open(int index)
    {
        _capture = new VideoCapture(index, VideoCaptureAPIs.DSHOW);
        _capture.Set(VideoCaptureProperties.FrameWidth, 1280);
        _capture.Set(VideoCaptureProperties.FrameHeight, 720);
        _capture.Set(VideoCaptureProperties.Fps, 30);
        if (!_capture.IsOpened())
            throw new InvalidOperationException("Не удалось открыть камеру. Закройте приложения, которые её используют.");
    }

    public Mat? Read()
    {
        var frame = new Mat();
        if (_capture?.Read(frame) != true || frame.Empty()) { frame.Dispose(); return null; }
        Cv2.Flip(frame, frame, FlipMode.Y);
        return frame;
    }

    public Mat Process(Mat input, bool correct, double strength)
    {
        var output = input.Clone();
        using var gray = new Mat();
        Cv2.CvtColor(input, gray, ColorConversionCodes.BGR2GRAY);
        var faces = _face.DetectMultiScale(gray, 1.12, 5, HaarDetectionTypes.ScaleImage, new Size(150, 150));
        LastEyesFound = 0;
        LastCorrectionMagnitude = 0;
        if (faces.Length == 0) return output;

        var face = faces.OrderByDescending(r => r.Width * r.Height).First();
        var upper = ClampRect(new Rect(face.X, face.Y + (int)(face.Height * .16), face.Width, (int)(face.Height * .43)), gray.Size());
        using var eyeBand = new Mat(gray, upper);
        var candidates = _eyes.DetectMultiScale(eyeBand, 1.08, 5, HaarDetectionTypes.ScaleImage, new Size(25, 18))
            .Select(r => new Rect(r.X + upper.X, r.Y + upper.Y, r.Width, r.Height)).ToArray();

        var mid = face.X + face.Width / 2;
        var left = candidates.Where(r => r.X + r.Width / 2 < mid).OrderByDescending(r => r.Width * r.Height).FirstOrDefault();
        var right = candidates.Where(r => r.X + r.Width / 2 >= mid).OrderByDescending(r => r.Width * r.Height).FirstOrDefault();
        if (left.Width == 0 || right.Width == 0) return output;

        var rawEyes = new[] { left, right };
        for (var i = 0; i < 2; i++)
        {
            _smoothEyes[i] = _hasSmoothEyes ? SmoothRect(_smoothEyes[i], rawEyes[i], .22) : rawEyes[i];
            var pupil = FindPupil(gray, _smoothEyes[i]);
            if (pupil is null) continue;
            var nx = (pupil.Value.X - _smoothEyes[i].X) / _smoothEyes[i].Width;
            var ny = (pupil.Value.Y - _smoothEyes[i].Y) / _smoothEyes[i].Height;
            if (nx is < .12 or > .88 || ny is < .18 or > .86) continue;

            // Low-pass filtering stops single dark eyelashes/reflections from making the iris jump.
            _smoothX[i] = _smoothX[i] * .78 + nx * .22;
            _smoothY[i] = _smoothY[i] * .82 + ny * .18;
            if (_calibrating) _samples[i].Add((_smoothX[i], _smoothY[i]));
            if (correct)
            {
                var magnitude = WarpIris(output, _smoothEyes[i], _smoothX[i], _smoothY[i], _targetX[i], _targetY[i], strength);
                LastCorrectionMagnitude = Math.Max(LastCorrectionMagnitude, magnitude);
            }
            LastEyesFound++;
        }
        _hasSmoothEyes = LastEyesFound == 2;
        return output;
    }

    private static Point2d? FindPupil(Mat gray, Rect eye)
    {
        var inner = ClampRect(new Rect(
            eye.X + (int)(eye.Width * .12), eye.Y + (int)(eye.Height * .24),
            (int)(eye.Width * .76), (int)(eye.Height * .58)), gray.Size());
        if (inner.Width < 8 || inner.Height < 6) return null;
        using var roi = new Mat(gray, inner);
        using var blur = new Mat();
        Cv2.GaussianBlur(roi, blur, new Size(5, 5), 0);
        Cv2.MeanStdDev(blur, out var mean, out var std);
        var cutoff = mean.Val0 - std.Val0 * .28;
        double wx = 0, wy = 0, total = 0;
        for (var y = 0; y < blur.Rows; y++)
        for (var x = 0; x < blur.Cols; x++)
        {
            var darkness = Math.Max(0, cutoff - blur.At<byte>(y, x));
            var centerBias = 1.0 - .35 * Math.Abs(x - blur.Cols / 2.0) / (blur.Cols / 2.0);
            var weight = darkness * darkness * centerBias;
            wx += x * weight; wy += y * weight; total += weight;
        }
        if (total < 1) return null;
        return new Point2d(inner.X + wx / total, inner.Y + wy / total);
    }

    private static double WarpIris(Mat frame, Rect eye, double px, double py, double tx, double ty, double strength)
    {
        var padX = (int)(eye.Width * .08); var padY = (int)(eye.Height * .08);
        var region = ClampRect(new Rect(eye.X - padX, eye.Y - padY, eye.Width + 2 * padX, eye.Height + 2 * padY), frame.Size());
        using var destination = new Mat(frame, region); using var source = destination.Clone();
        using var mapX = new Mat(region.Height, region.Width, MatType.CV_32FC1);
        using var mapY = new Mat(region.Height, region.Width, MatType.CV_32FC1);
        // Strong mode: enough travel to make reading above/below the lens visibly redirect to the calibrated point.
        var dx = Math.Clamp((tx - px) * eye.Width * strength * 1.65, -eye.Width * .30, eye.Width * .30);
        var dy = Math.Clamp((ty - py) * eye.Height * strength * 1.55, -eye.Height * .24, eye.Height * .24);
        var centerX = eye.X - region.X + tx * eye.Width;
        var centerY = eye.Y - region.Y + ty * eye.Height;
        var sigmaX = Math.Max(3, eye.Width * .27); var sigmaY = Math.Max(2, eye.Height * .32);
        for (var y = 0; y < region.Height; y++)
        for (var x = 0; x < region.Width; x++)
        {
            var weight = Math.Exp(-.5 * (Math.Pow((x - centerX) / sigmaX, 2) + Math.Pow((y - centerY) / sigmaY, 2)));
            mapX.Set(y, x, (float)(x - dx * weight)); mapY.Set(y, x, (float)(y - dy * weight));
        }
        Cv2.Remap(source, destination, mapX, mapY, InterpolationFlags.Cubic, BorderTypes.Reflect101);
        return Math.Sqrt(dx * dx + dy * dy);
    }

    private static Rect SmoothRect(Rect old, Rect current, double a) => new(
        (int)(old.X * (1-a) + current.X * a), (int)(old.Y * (1-a) + current.Y * a),
        Math.Max(1, (int)(old.Width * (1-a) + current.Width * a)), Math.Max(1, (int)(old.Height * (1-a) + current.Height * a)));
    private static Rect ClampRect(Rect r, Size s)
    {
        var x = Math.Clamp(r.X, 0, s.Width - 1); var y = Math.Clamp(r.Y, 0, s.Height - 1);
        return new Rect(x, y, Math.Max(1, Math.Min(r.Width, s.Width - x)), Math.Max(1, Math.Min(r.Height, s.Height - y)));
    }

    public void BeginCalibration() { foreach (var list in _samples) list.Clear(); _calibrating = true; }
    public void SaveCalibration()
    {
        _calibrating = false; if (_samples.Any(s => s.Count < 10)) return;
        for (var i = 0; i < 2; i++) { _targetX[i] = _samples[i].Average(s => s.X); _targetY[i] = _samples[i].Average(s => s.Y); }
        Directory.CreateDirectory(Path.GetDirectoryName(ProfilePath)!);
        File.WriteAllText(ProfilePath, JsonSerializer.Serialize(new { LeftX=_targetX[0], LeftY=_targetY[0], RightX=_targetX[1], RightY=_targetY[1], Created=DateTimeOffset.Now }));
        _profileLoaded = true;
    }
    private void LoadCalibration()
    {
        try
        {
            if (!File.Exists(ProfilePath)) return; using var doc = JsonDocument.Parse(File.ReadAllText(ProfilePath)); var r = doc.RootElement;
            if (r.TryGetProperty("LeftX", out var lx)) { _targetX[0]=lx.GetDouble(); _targetY[0]=r.GetProperty("LeftY").GetDouble(); _targetX[1]=r.GetProperty("RightX").GetDouble(); _targetY[1]=r.GetProperty("RightY").GetDouble(); _profileLoaded=true; }
        }
        catch { /* damaged/old profile falls back to safe central targets */ }
    }
    public void Close() { _capture?.Release(); _capture?.Dispose(); _capture=null; _hasSmoothEyes=false; }
    public void Dispose() { Close(); _face.Dispose(); _eyes.Dispose(); }
}
