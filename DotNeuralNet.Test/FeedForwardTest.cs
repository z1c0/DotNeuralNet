using System.Linq;
using Xunit;

namespace DotNeuralNet.Test
{
  public class FeedForwardTest
  {
    [Fact]
    public void Test_SimpleNetwork()
    {
      var network = new Network(3, 2, 1);
      // Input nodes.
      network.InputNodes[0].Value = 0.1;
      network.InputNodes[1].Value = 0.2;
      network.InputNodes[2].Value = 0.3;
      // Configure weights and bias for hidden node.
      network.HiddenNodes[0].Bias = 0.2;
      network.HiddenNodes[0].Incoming[0].Weight = 0.4;
      network.HiddenNodes[0].Incoming[1].Weight = 0.5;
      network.HiddenNodes[0].Incoming[2].Weight = 0.6;
      network.HiddenNodes[1].Bias = 0.8;
      network.HiddenNodes[1].Incoming[0].Weight = 0.7;
      network.HiddenNodes[1].Incoming[1].Weight = 0.6;
      network.HiddenNodes[1].Incoming[2].Weight = 0.5;
      // Confgure weights and bias for output node.
      network.OutputNodes[0].Bias = 0.5;
      network.OutputNodes[0].Incoming[0].Weight = 0.3;
      network.OutputNodes[0].Incoming[1].Weight = 0.4;

      Assert.Equal(0.72, network.OutputNodes[0].Value, 2);
    }
  }
}
