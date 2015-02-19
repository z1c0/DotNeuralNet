using System.Linq;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DotNeuralNet.Test
{
  [TestClass]
  public class NetworkTest
  {
    [TestMethod]
    public void TestMethod1()
    {
      var network = new Network(5, 4, 2);
      foreach (var i in network.InputNodes)
      {
        i.Value = 3;
      }
      Assert.AreEqual(3, network.OutputNodes.ElementAt(0).Value);
      Assert.AreEqual(4, network.OutputNodes.ElementAt(1).Value);
    }
  }
}
