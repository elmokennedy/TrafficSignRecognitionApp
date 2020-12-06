using System.Collections.Generic;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;

namespace TrafficSignRecognitionApp.Custom.DataStructures
{
    public struct ClassificationImageSettings
    {
        public const int imageHeight = 224;
        public const int imageWidth = 224;
    }

    public class TrafficSignClassification
    {
        [ColumnName("classLabel")]
        public string[] Labels;

        [ColumnName("loss")]
        [OnnxSequenceType(typeof(IDictionary<string, float>))]
        public IEnumerable<IDictionary<string, float>> Scores { get; set; }
    }
}
