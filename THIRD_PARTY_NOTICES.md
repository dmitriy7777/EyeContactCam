# Third-party notices

EyeContactCam uses OpenCvSharp, Microsoft ONNX Runtime and two converted gaze-redirection ONNX model files (`gaze_L.onnx`, `gaze_R.onnx`).

The model architecture and weights originate from the research implementation “Look at Me! Correcting Eye Gaze in Live Video Communication” by Chih-Fan Hsu et al.:

- https://github.com/chihfanhsu/gaze_correction
- https://doi.org/10.1145/3311784

The ONNX conversion used for this local prototype was obtained from:

- https://github.com/KypMon/coreml-eye-contact

That conversion repository did not contain an explicit license file when inspected. The copied ONNX files are therefore included here only for local evaluation/prototyping. Confirm redistribution rights before publishing or selling a binary containing them.
