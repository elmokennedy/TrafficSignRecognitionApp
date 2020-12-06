using Microsoft.ML.Data;

namespace TrafficSignRecognitionApp.ObjectDetection.DataStructures
{
    public class DetectionPredictions
    {
        [ColumnName("model_outputs0")]
        public float[] PredictedLabels { get; set; }
    }
}
