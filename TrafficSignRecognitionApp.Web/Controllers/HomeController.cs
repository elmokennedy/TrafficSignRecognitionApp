using System;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Net;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using TrafficSignRecognitionApp.ObjectDetection.TrafficSignDetection;
using TrafficSignRecognitionApp.Web.Models;

namespace TrafficSignRecognitionApp.Web.Controllers
{
    public class HomeController : Controller
    {
        private const string ModelPathName = "DetectionModel2";

        private readonly ITrafficSignDetectionOutputParser _trafficSignDetectionOutputParser;

        public HomeController(ITrafficSignDetectionOutputParser trafficSignDetectionOutputParser)
        {
            _trafficSignDetectionOutputParser = trafficSignDetectionOutputParser;
        }

        public IActionResult Index()
        {
            return View();
        }

        public async Task<IActionResult> UploadData(IFormFile image, int threshold)
        {
            try
            {
                using (var memoryStream = new MemoryStream())
                {
                    await image.CopyToAsync(memoryStream);
                    using (var img = (Bitmap)Image.FromStream(memoryStream))
                    {
                        var trafficSignOutput = _trafficSignDetectionOutputParser.ProcessTheImage(
                            img, 
                            threshold,
                            ModelPathName);

                        var imageBase64Data = Convert.ToBase64String(trafficSignOutput.PredictedImageByteData);
                        var predictedImg = $"data:image/jpeg;base64,{imageBase64Data}";

                        return Json(new { predictedImg, predictions = trafficSignOutput.Predictions, success = true });
                    }
                }
            }
            catch (Exception ex)
            {
                return Json(new { error = ex.Message, success = false });
            }
        }

        public IActionResult UploadUrlData(string imageUrl, int threshold)
        {
            try
            {
                using (var wc = new WebClient())
                {
                    using (var stream = wc.OpenRead(imageUrl))
                    {
                        using (var img = new Bitmap(stream))
                        {
                            var trafficSignOutput = _trafficSignDetectionOutputParser.ProcessTheImage(
                                img, 
                                threshold,
                                ModelPathName);

                            var imageBase64Data = Convert.ToBase64String(trafficSignOutput.PredictedImageByteData);
                            var predictedImg = $"data:image/jpeg;base64,{imageBase64Data}";

                            return Json(new { predictedImg, predictions = trafficSignOutput.Predictions, success = true });
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                return Json(new { error = ex.Message, success = false });
            }
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
