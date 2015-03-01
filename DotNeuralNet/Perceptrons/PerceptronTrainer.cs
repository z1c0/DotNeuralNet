using System;
using System.Collections.Generic;
using System.Linq;

namespace DotNeuralNet.Perceptrons
{
  public class PerceptronTrainer
  {
    private readonly Perceptron _perceptron;

    public PerceptronTrainer(Perceptron perceptron)
    {
      _perceptron = perceptron;
    }

    public void Train(IEnumerable<PerceptronTrainingRow> rows, double adjust, int rounds)
    {
      for (var r = 0; r < rounds; r++)
      {
        foreach (var row in rows)
        {
          RandomizeOrder(row.Inputs);
          for (var i = 0; i < row.Inputs.Length; i++)
          {
            _perceptron.Inputs[i] = row.Inputs[i];
          }
          var diff = _perceptron.Output - row.ExpectedOutput;
          var delta = adjust * Math.Abs(diff) * Math.Sign(diff);
          if (Math.Abs(diff) > 0)
          {
            for (var i = 0; i < _perceptron.Weights.Length; i++)
            {
              _perceptron.Weights[i] -= delta;
            }
            _perceptron.Bias -= delta;
          }
        }
      }
    }

    private void RandomizeOrder(double[] inputs)
    {
      for (var i = 0; i < inputs.Length; i++)
      {
        var j = _perceptron.Random.Next(i, inputs.Length);
        var tmp = inputs[i];
        inputs[i] = inputs[j];
        inputs[j] = tmp;
      }
    }
  }
}
