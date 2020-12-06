namespace TrafficSignRecognitionApp.ObjectDetection.Models
{
    public class TrafficSignPrediction
    {
        public string Tag { get; set; }

        public float Probability { get; set; }

        public string ImagePath { get; set; }
    }
}
