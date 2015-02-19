using System;

namespace DotNeuralNet
{
  internal static class Helpers
  {
    private readonly static Random _random = new Random();

    public static double GetRandomWeight()
    {
      return _random.NextDouble();
    }
  }
}
