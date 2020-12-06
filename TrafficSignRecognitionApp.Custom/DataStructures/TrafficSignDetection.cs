using Microsoft.ML.Data;

namespace TrafficSignRecognitionApp.Custom.DataStructures
{
    public struct DetectionImageSettings
    {
        public const int imageHeight = 416;
        public const int imageWidth = 416;
    }
      
    public class TrafficSignDetection
    {
        [ColumnName("model_outputs0")]
        public float[] PredictedLabels { get; set; }
    }
}
