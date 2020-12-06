using System.Collections.Generic;

namespace TrafficSignRecognitionApp.ObjectDetection.Models
{
    public class TrafficSignOutput
    {
        public TrafficSignOutput()
        {
            Predictions = new List<TrafficSignPrediction>();
        }

        public byte[] PredictedImageByteData { get; set; }

        public IEnumerable<TrafficSignPrediction> Predictions { get; set; }
    }
}
