using System.Drawing;
using Microsoft.ML.Transforms.Image;

namespace TrafficSignRecognitionApp.Custom.DataStructures
{
    public class TrafficSignInput
    {
        [ImageType(DetectionImageSettings.imageHeight, DetectionImageSettings.imageWidth)]
        public Bitmap Image { get; set; }
    }
}
