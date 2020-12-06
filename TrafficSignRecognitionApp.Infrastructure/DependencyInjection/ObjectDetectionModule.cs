using Autofac;
using TrafficSignRecognitionApp.ObjectDetection.TrafficSignDetection;

namespace TrafficSignRecognitionApp.Infrastructure.DependencyInjection
{
    public class ObjectDetectionModule : Module
    {
        protected override void Load(ContainerBuilder builder)
        {
            builder.RegisterType<TrafficSignDetectionOutputParser>()
                .As<ITrafficSignDetectionOutputParser>()
                .InstancePerLifetimeScope();
        }
    }
}
