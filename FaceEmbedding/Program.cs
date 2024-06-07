using Emgu.CV;
using Emgu.CV.Face;
using Emgu.CV.Ocl;
using Emgu.CV.Structure;
using Emgu.CV.Util;

using FaceAiSharp;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
var det = FaceAiSharpBundleFactory.CreateFaceDetector();
var rec = FaceAiSharpBundleFactory.CreateFaceEmbeddingsGenerator();
var haarCascade = new CascadeClassifier("models/haarcascade_frontalface_default.xml");
var frec = new FaceRecognizerSF("models/recognition.onnx", "", 0, 0);
FacemarkLBFParams fParams = new FacemarkLBFParams();
fParams.ModelFile = @"models\lbfmodel.yaml";
fParams.NLandmarks = 68; // number of landmark points
fParams.InitShapeN = 10; // number of multiplier for make data augmentation
fParams.StagesN = 5; // amount of refinement stages
fParams.TreeN = 6; // number of tree in the model for each landmark point
fParams.TreeDepth = 5; //he depth of decision tree
FacemarkLBF facemark = new FacemarkLBF(fParams);
A:
Console.WriteLine("Enter the path to the image you want to analyze:");
var input = Console.ReadLine();
if (input.Split(' ').Length == 1)
{
    var path = input.Split(' ')[0].Trim('"');
    var image = Image.Load<Rgb24>(path);
    var now = DateTime.Now;
    var faces = det.DetectFaces(image);
    Console.WriteLine($"Detected {faces.Count} faces in {(DateTime.Now - now).Milliseconds}");
    foreach (var face in faces)
    {
        now = DateTime.Now;
        using var timg = image.Clone();
        rec.AlignFaceUsingLandmarks(timg, face.Landmarks!);
        var embedding = rec.GenerateEmbedding(timg);
        Console.WriteLine($"Face embedding generated in {(DateTime.Now - now).Milliseconds}");
    }
    image.Dispose();
}
if (input.Split(' ').Length == 2 && input.Split(' ')[1] == "-er")
{
    //face embeddings using emgucv
    var path = input.Split(' ')[0].Trim('"');
    //emgucv detector

    var image = CvInvoke.Imread(path);
    Image<Gray, byte> grayframe = image.ToImage<Gray, byte>();
    var now = DateTime.Now;
    VectorOfRect faces = new VectorOfRect(haarCascade.DetectMultiScale(grayframe, 1.2, 10, new System.Drawing.Size(50, 50)));

    Console.WriteLine($"Detected {faces.Size} faces in {(DateTime.Now - now).Milliseconds}");
    VectorOfVectorOfPointF landmarks = new VectorOfVectorOfPointF();
    facemark.LoadModel(fParams.ModelFile);

    bool success = facemark.Fit(grayframe, faces, landmarks);

    for (int i = 0; i < faces.Size; i++)
    {
        now = DateTime.Now;
        var aligned = new Mat();
        var embedding = new Mat();
        var faceBox = new Emgu.CV.Matrix<float>(1, 14);
        var keypoints = new List<float>();
        var xembedding = new List<float>();


        List<System.Drawing.PointF> leftEyeCenter = new();

        for (int n = 36; n < 42; n++)
        {
            leftEyeCenter.Add(landmarks[i][n]);
        }

        float leftEyeCenterX = 0;
        float leftEyeCenterY = 0;
        foreach (var pointa in leftEyeCenter)
        {
            leftEyeCenterX += pointa.X;
            leftEyeCenterY += pointa.Y;
        }
        leftEyeCenterX /= leftEyeCenter.Count;
        leftEyeCenterY /= leftEyeCenter.Count;

        List<System.Drawing.PointF> rightEyeCenter = new();
        for (int n = 42; n < 48; n++)
        {
            rightEyeCenter.Add(landmarks[i][n]);
        }

        float rightEyeCenterX = 0;
        float rightEyeCenterY = 0;
        foreach (var pointb in rightEyeCenter)
        {
            rightEyeCenterX += pointb.X;
            rightEyeCenterY += pointb.Y;
        }
        rightEyeCenterX /= rightEyeCenter.Count;
        rightEyeCenterY /= rightEyeCenter.Count;

        faceBox[0, 4] = leftEyeCenterX + 0f;
        faceBox[0, 5] = leftEyeCenterY + 0f;
        faceBox[0, 6] = rightEyeCenterX + 0f;
        faceBox[0, 7] = rightEyeCenterY + 0f;
        faceBox[0, 8] = landmarks[i][30].X + 0f;
        faceBox[0, 9] = landmarks[i][30].Y + 0f;
        faceBox[0, 10] = landmarks[i][48].X + 0f;
        faceBox[0, 11] = landmarks[i][48].Y + 0f;
        faceBox[0, 12] = landmarks[i][54].X + 0f;
        faceBox[0, 13] = landmarks[i][54].Y + 0f;
        for (int k = 0; k < 10; k++)
        {

            keypoints.Add(faceBox[0, k + 4]);
        }

        frec.AlignCrop(image, faceBox, aligned);
        frec.Feature(aligned, embedding);
        Console.WriteLine($"Face embedding generated in {(DateTime.Now - now).Milliseconds}");
    }
    //var now = DateTime.Now;

    image.Dispose();
}
if (input.Split(' ').Length == 2 && input.Split(' ')[1] == "-d")
{
    var path = input.Split(' ')[0].Trim('"');
    var image = Image.Load<Rgb24>(path);
    var now = DateTime.Now;
    for (int i = 0; i < 100; i++)
    {
        det.DetectFaces(image);
    }
    image.Dispose();
}
if (input.Split(' ').Length == 2 && input.Split(' ')[1] == "-r")
{
    var path = input.Split(' ')[0].Trim('"');
    var image = Image.Load<Rgb24>(path);
    var now = DateTime.Now;
    for (int i = 0; i < 100; i++)
    {
        rec.GenerateEmbedding(image);
    }
    image.Dispose();
}
Console.WriteLine("-----------------------------");
goto A;
