using System;

namespace DotNeuralNet
{
  public class OutputNode : WeightedNode
  {
    public OutputNode()
    {
      _activationFunction = Sigmoid;
    }

    private static double Sigmoid(double value)
    {
      if (value < -45.0) return 0.0;
      else if (value > 45.0) return 1.0;

      return 1.0 / (1.0 + Math.Exp(-value));
    }

    public double BiasDelta { get; set; }
  }
}
