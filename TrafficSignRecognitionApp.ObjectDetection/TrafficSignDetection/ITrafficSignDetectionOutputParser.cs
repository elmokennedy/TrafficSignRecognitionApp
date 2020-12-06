using System.Drawing;
using TrafficSignRecognitionApp.ObjectDetection.Models;

namespace TrafficSignRecognitionApp.ObjectDetection.TrafficSignDetection
{
    public interface ITrafficSignDetectionOutputParser
    {
        TrafficSignOutput ProcessTheImage(Bitmap image, int threshold, string modelFolder);
    }
}
