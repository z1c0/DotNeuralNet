using System;
using System.Linq;

namespace DotNeuralNet.Perceptrons
{
  public class Perceptron
  {
    public Perceptron(int numberOfInputs)
    {
      Inputs = new double[numberOfInputs];

      Random = new Random(numberOfInputs);
      Weights = Enumerable.Range(0, numberOfInputs).Select(_ => RandomWeight()).ToArray();
      Bias = RandomWeight();
    }

    internal Random Random { get; private set; }

    private double RandomWeight()
    {
      return (-1 + 2 * Random.NextDouble()) / 10;
    }

    public double[] Inputs { get; private set; }

    public double[] Weights { get; private set; }

    public double Bias { get; set; }

    public Func<double, double> ActivationFunction { get; set; }

    public double Output
    {
      get
      {
        var s = Bias;
        for (var i = 0; i < Inputs.Length; i++)
        {
          s += Inputs[i] * Weights[i];
        }
        return ActivationFunction(s);
      }
    }
  }
}
