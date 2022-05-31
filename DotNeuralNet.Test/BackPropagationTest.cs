using System.Linq;
using System.Collections.Generic;
using Xunit;

namespace DotNeuralNet.Test
{
  public class BackPropagationTest
  {
    [Fact]
    public void CheckOutputs_NoTraining()
    {
      var network = new Network(3, 4, 2);

      network.InputNodes[0].Value = 1.0;
      network.InputNodes[1].Value = -2.0;
      network.InputNodes[2].Value = 3.0;

      Helper_InitWeights(network);

      Assert.Equal(0.50696673, network.OutputNodes[0].Value, 8);
      Assert.Equal(0.50725216, network.OutputNodes[1].Value, 8);
    }

    [Fact]
    public void Test_Train_SimpleRow()
    {
      var network = new Network(3, 4, 2);

      Helper_InitWeights(network);

      var trainer = new BackPropagationTrainer(network);
      var rows = new[] { new BackPropagationTrainingRow (new [] { 1.0, -2.0, 3.0 }, new [] { 0.1234, 0.8766 }) };
      trainer.Train(rows, 0.5, 10000);

      const int tolerance = 4;
      Assert.Equal(0.07770, network.HiddenNodes[0].Incoming[0].Weight, tolerance);
      Assert.Equal(0.08118, network.HiddenNodes[1].Incoming[0].Weight, tolerance);
      Assert.Equal(0.08441, network.HiddenNodes[2].Incoming[0].Weight, tolerance);

      network.Invalidate();
      for (var i = 0; i < rows[0].Inputs.Length; i++)
      {
        network.InputNodes[i].Value = rows[0].Inputs[i];
      }
      Assert.Equal(rows[0].Outputs[0], network.OutputNodes[0].Value, tolerance);
      Assert.Equal(rows[0].Outputs[1], network.OutputNodes[1].Value, tolerance);
    }

    [Fact]
    public void Test_LargeNet_ManyRows()
    {
      //var network = new Network(10 * 10, 8 * 8, 10);

      //var trainer = new BackPropagationTrainer(network);
      //var rows = new List<BackPropagationTrainingRow>();
      //for (var i = 0; i < 200; i++)
      //{
      //  rows.Add(new BackPropagationTrainingRow(new double[100], new double[10]));
      //}
      //trainer.Train(rows, 0.5, 100);
    }

    private static void Helper_InitWeights(Network network)
    {
      //
      // Init weights to have deterministic test.
      //
      network.HiddenNodes[0].Incoming[0].Weight = 0.001;
      network.HiddenNodes[0].Incoming[1].Weight = 0.005;
      network.HiddenNodes[0].Incoming[2].Weight = 0.009;

      network.HiddenNodes[1].Incoming[0].Weight = 0.002;
      network.HiddenNodes[1].Incoming[1].Weight = 0.006;
      network.HiddenNodes[1].Incoming[2].Weight = 0.010;

      network.HiddenNodes[2].Incoming[0].Weight = 0.003;
      network.HiddenNodes[2].Incoming[1].Weight = 0.007;
      network.HiddenNodes[2].Incoming[2].Weight = 0.011;

      network.HiddenNodes[3].Incoming[0].Weight = 0.004;
      network.HiddenNodes[3].Incoming[1].Weight = 0.008;
      network.HiddenNodes[3].Incoming[2].Weight = 0.012;

      network.HiddenNodes[0].Bias = 0.013;
      network.HiddenNodes[1].Bias = 0.014;
      network.HiddenNodes[2].Bias = 0.015;
      network.HiddenNodes[3].Bias = 0.016;

      network.OutputNodes[0].Incoming[0].Weight = 0.017;
      network.OutputNodes[0].Incoming[1].Weight = 0.019;
      network.OutputNodes[0].Incoming[2].Weight = 0.021;
      network.OutputNodes[0].Incoming[3].Weight = 0.023;

      network.OutputNodes[1].Incoming[0].Weight = 0.018;
      network.OutputNodes[1].Incoming[1].Weight = 0.020;
      network.OutputNodes[1].Incoming[2].Weight = 0.022;
      network.OutputNodes[1].Incoming[3].Weight = 0.024;

      network.OutputNodes[0].Bias = 0.025;
      network.OutputNodes[1].Bias = 0.026;
    }
  }
}
