using System.Drawing;

namespace TrafficSignRecognitionApp.ObjectDetection.BoundingBoxModels
{
    public class BoundingBox
    {
        public BoundingBoxDimensions Dimensions { get; set; }

        public string Label { get; set; }

        public float Confidence { get; set; }

        public RectangleF Rect => new RectangleF(Dimensions.X, Dimensions.Y, Dimensions.Width, Dimensions.Height);

        public Color BoxColor { get; set; }

        public string Description => $"{Label} ({Confidence * 100:0}%)";
    }
}
