using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Transforms.Image;
using TrafficSignRecognitionApp.ObjectDetection.BoundingBoxModels;
using TrafficSignRecognitionApp.ObjectDetection.DataStructures;
using TrafficSignRecognitionApp.ObjectDetection.Models;
using TrafficSignRecognitionApp.ObjectDetection.SettingsStructs;

namespace TrafficSignRecognitionApp.ObjectDetection.TrafficSignDetection
{
    public class TrafficSignDetectionOutputParser: ITrafficSignDetectionOutputParser
    {
        public const int rowCount = 13, columnCount = 13;

        public const int featuresPerBox = 5;

        private static readonly (float x, float y)[] boxAnchors = { (0.573f, 0.677f), (1.87f, 2.06f), (3.34f, 5.47f), (7.88f, 3.53f), (9.77f, 9.17f) };

        public TrafficSignOutput ProcessTheImage(Bitmap image, int threshold, string modelFolder)
        {
            var context = new MLContext();

            var model = LoadModel(context, modelFolder);

            var predictionEngine = context.Model.CreatePredictionEngine<DetectionInput, DetectionPredictions>(model);

            var detectionPrediction = predictionEngine.Predict(new DetectionInput { Image = image });

            var floatThreshold = (float)threshold/100;

            var labels = File.ReadAllLines(Path.Combine(modelFolder, "labels.txt"));

            var boundingBoxes = ParseOutputs(detectionPrediction.PredictedLabels, labels, floatThreshold);

            var originalWidth = image.Width;
            var originalHeight = image.Height;

            var trafficSignPredictions = new List<TrafficSignPrediction>();

            var uniqueBoundingBoxes = boundingBoxes
                .GroupBy(x => x.Label)
                .Select(grp =>
                {
                    var confidence = grp
                        .OrderByDescending(x => x.Confidence)
                        .Select(x => x.Confidence)
                        .FirstOrDefault();

                    return new BoundingBox
                    {
                        Label = grp.Key,
                        Confidence = confidence,
                        Dimensions = grp.FirstOrDefault(x => x.Confidence == confidence).Dimensions
                    };
                }).ToList();

            foreach (var boundingBox in uniqueBoundingBoxes)
            {
                var x = Math.Max(boundingBox.Dimensions.X, 0);
                var y = Math.Max(boundingBox.Dimensions.Y, 0);
                var width = Math.Min(originalWidth - x, boundingBox.Dimensions.Width);
                var height = Math.Min(originalHeight - y, boundingBox.Dimensions.Height);

                // fit to current image size
                x = originalWidth * x / ImageDetectionSettings.ImageWidth;
                y = originalHeight * y / ImageDetectionSettings.ImageHeight;
                width = originalWidth * width / ImageDetectionSettings.ImageWidth;
                height = originalHeight * height / ImageDetectionSettings.ImageHeight;

                using (var graphics = Graphics.FromImage(image))
                {
                    graphics.DrawRectangle(new Pen(Color.Red, 3), x, y, width, height);
                    graphics.DrawString(boundingBox.Description, new Font(FontFamily.Families[0], 30f), Brushes.Red, x + 5, y + 5);
                }

                trafficSignPredictions.Add(new TrafficSignPrediction
                {
                    Tag = boundingBox.Label,
                    Probability = boundingBox.Confidence * 100,
                    ImagePath = GetTagImage(boundingBox.Label)
                });
            }

            using (var stream = new MemoryStream())
            {
                image.Save(stream, System.Drawing.Imaging.ImageFormat.Jpeg);
                return new TrafficSignOutput
                {
                    PredictedImageByteData = stream.ToArray(),
                    Predictions = trafficSignPredictions
                };
            }
        }

        private ITransformer LoadModel(MLContext context, string modelFolder)
        {
            var emptyData = new List<DetectionInput>();

            var data = context.Data.LoadFromEnumerable(emptyData);

            var pipeline = context.Transforms.ResizeImages(
                    resizing: ImageResizingEstimator.ResizingKind.Fill,
                    outputColumnName: "data",
                    imageWidth: ImageDetectionSettings.ImageWidth,
                    imageHeight: ImageDetectionSettings.ImageHeight,
                    inputColumnName: nameof(DetectionInput.Image))
                .Append(context.Transforms.ExtractPixels(outputColumnName: "data"))
                .Append(context.Transforms.ApplyOnnxModel(
                    modelFile: Path.Combine(modelFolder, "model.onnx"),
                    outputColumnName: "model_outputs0",
                    inputColumnName: "data"));

            return pipeline.Fit(data);
        }

        private static List<BoundingBox> ParseOutputs(float[] modelOutput, string[] labels, float probabilityThreshold = .5f)
        {
            var boxes = new List<BoundingBox>();

            for (var row = 0; row < rowCount; row++)
            {
                for (var column = 0; column < columnCount; column++)
                {
                    for (var box = 0; box < boxAnchors.Length; box++)
                    {
                        var channel = box * (labels.Length + featuresPerBox);

                        var boundingBoxPrediction = ExtractBoundingBoxPrediction(modelOutput, row, column, channel);

                        var mappedBoundingBox = MapBoundingBoxToCell(row, column, box, boundingBoxPrediction);

                        if (boundingBoxPrediction.Confidence < probabilityThreshold)
                            continue;

                        var classProbabilities = ExtractClassProbabilities(modelOutput, row, column, channel, boundingBoxPrediction.Confidence, labels);

                        var (topProbability, topIndex) = classProbabilities.Select((probability, index) => (Score: probability, Index: index)).Max();

                        if (topProbability < probabilityThreshold)
                            continue;

                        boxes.Add(new BoundingBox
                        {
                            Dimensions = mappedBoundingBox,
                            Confidence = topProbability,
                            Label = labels[topIndex]
                        });
                    }
                }
            }

            return boxes;
        }

        private static BoundingBoxDimensions MapBoundingBoxToCell(int row, int column, int box, BoundingBoxPrediction boxDimensions)
        {
            const float cellWidth = ImageDetectionSettings.ImageWidth / columnCount;
            const float cellHeight = ImageDetectionSettings.ImageHeight / rowCount;

            var mappedBox = new BoundingBoxDimensions
            {
                X = (row + Sigmoid(boxDimensions.X)) * cellWidth,
                Y = (column + Sigmoid(boxDimensions.Y)) * cellHeight,
                Width = MathF.Exp(boxDimensions.Width) * cellWidth * boxAnchors[box].x,
                Height = MathF.Exp(boxDimensions.Height) * cellHeight * boxAnchors[box].y,
            };

            // The x,y coordinates from the (mapped) bounding box prediction represent the center
            // of the bounding box. We adjust them here to represent the top left corner.
            mappedBox.X -= mappedBox.Width / 2;
            mappedBox.Y -= mappedBox.Height / 2;

            return mappedBox;
        }

        private static BoundingBoxPrediction ExtractBoundingBoxPrediction(float[] modelOutput, int row, int column, int channel)
        {
            return new BoundingBoxPrediction
            {
                X = modelOutput[GetOffset(row, column, channel++)],
                Y = modelOutput[GetOffset(row, column, channel++)],
                Width = modelOutput[GetOffset(row, column, channel++)],
                Height = modelOutput[GetOffset(row, column, channel++)],
                Confidence = Sigmoid(modelOutput[GetOffset(row, column, channel++)])
            };
        }

        public static float[] ExtractClassProbabilities(float[] modelOutput, int row, int column, int channel, float confidence, string[] labels)
        {
            var classProbabilitiesOffset = channel + featuresPerBox;
            var classProbabilities = new float[labels.Length];
            for (var classProbability = 0; classProbability < labels.Length; classProbability++)
                classProbabilities[classProbability] = modelOutput[GetOffset(row, column, classProbability + classProbabilitiesOffset)];
            return Softmax(classProbabilities).Select(p => p * confidence).ToArray();
        }

        private static float Sigmoid(float value)
        {
            var k = MathF.Exp(value);
            return k / (1.0f + k);
        }

        private static float[] Softmax(float[] classProbabilities)
        {
            var max = classProbabilities.Max();
            var exp = classProbabilities.Select(v => MathF.Exp(v - max));
            var sum = exp.Sum();
            return exp.Select(v => v / sum).ToArray();
        }

        private static int GetOffset(int row, int column, int channel)
        {
            const int channelStride = rowCount * columnCount;
            return (channel * channelStride) + (column * columnCount) + row;
        }

        private string GetTagImage(string tag)
        {
            switch (tag)
            {
                case "Main road":
                    return @"images\traffic signs\main_road.jpg";
                case "Pedestrian crossing":
                    return @"images\traffic signs\pedestrian_crossing.png";
                case "Stop":
                    return @"images\traffic signs\stop.png";
                case "Yield":
                    return @"images\traffic signs\yield.png";
                default:
                    return tag;
            }
        }
    }
}
