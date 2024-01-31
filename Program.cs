// See https://aka.ms/new-console-template for more information
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Face;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;

class Program
{
    static void Main()
    {
        // Rutas del video y la carpeta de salida
        string videoPath = @"C:\Proyecto\MiProyectoCSharp\Video.mp4";
        string outputFolder = @"C:\Proyecto\MiProyectoCSharp\Frames";

        // Crea el clasificador Haar para detección de rostros
        CascadeClassifier faceCascade = new CascadeClassifier("haarcascade_frontalface_default.xml");

        // Crea el reconocedor de rostros
        FaceRecognizer recognizer = new EigenFaceRecognizer(80, double.PositiveInfinity);

        // Carga el modelo entrenado
        recognizer.Read("trained_model.yml");

        // Abre el video
        VideoCapture videoCapture = new VideoCapture(videoPath);

        if (!videoCapture.IsOpened)
        {
            Console.WriteLine("No se pudo abrir el video.");
            return;
        }

        // Crea la carpeta de salida si no existe
        if (!Directory.Exists(outputFolder))
            Directory.CreateDirectory(outputFolder);

        // Variables para el procesamiento de frames
        int frameCount = (int)videoCapture.Get(CapProp.FrameCount);
        int frameRate = (int)videoCapture.Get(CapProp.Fps);
        int totalFrames = frameRate * 10; // Limita el procesamiento a 10 segundos

        // Diccionario para almacenar las imágenes de cada persona
        Dictionary<int, Mat> personImages = new Dictionary<int, Mat>();

        // Procesa cada frame del video
        for (int i = 0; i < totalFrames; i++)
        {
            Mat frame = new Mat();
            videoCapture.Read(frame);

            if (frame.IsEmpty)
            {
                Console.WriteLine("Fin del video.");
                break;
            }

            // Convierte la imagen a escala de grises
            Mat grayFrame = new Mat();
            CvInvoke.CvtColor(frame, grayFrame, ColorConversion.Bgr2Gray);

            // Detecta rostros en la imagen
            Rectangle[] faces = faceCascade.DetectMultiScale(grayFrame, 1.1, 3, Size.Empty);

            // Dibuja rectángulos alrededor de los rostros detectados
            foreach (Rectangle face in faces)
            {
                CvInvoke.Rectangle(frame, face, new MCvScalar(0, 255, 0), 2);

                // Extrae la región de interés (ROI) que contiene el rostro
                Mat faceROI = new Mat(grayFrame, face);

                // Realiza la predicción para el rostro
                int predictedLabel = recognizer.Predict(faceROI).Label;

                // Almacena el frame actual en el diccionario
                if (!personImages.ContainsKey(predictedLabel))
                {
                    personImages[predictedLabel] = frame.Clone();
                }
                else
                {
                    // Compara el frame actual con el frame almacenado y guarda la diferencia
                    Mat difference = new Mat();
                    CvInvoke.AbsDiff(personImages[predictedLabel], frame, difference);

                    // Guarda la diferencia facial como una imagen en la carpeta de salida
                    string outputFileName = Path.Combine(outputFolder, $"difference_{predictedLabel}_{i}.jpg");
                    CvInvoke.Imwrite(outputFileName, difference);
                }
            }
        }

        // Libera los recursos
        videoCapture.Dispose();
        CvInvoke.DestroyAllWindows();
    }
}