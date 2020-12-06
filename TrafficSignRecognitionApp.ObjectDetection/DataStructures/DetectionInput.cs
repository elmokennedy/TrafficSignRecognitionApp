using System.Drawing;
using Microsoft.ML.Transforms.Image;
using TrafficSignRecognitionApp.ObjectDetection.SettingsStructs;

namespace TrafficSignRecognitionApp.ObjectDetection.DataStructures
{
    public class DetectionInput
    {
        [ImageType(ImageDetectionSettings.ImageHeight, ImageDetectionSettings.ImageWidth)]
        public Bitmap Image { get; set; }
    }
}
