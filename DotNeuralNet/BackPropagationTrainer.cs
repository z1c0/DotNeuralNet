using System;
using System.Collections.Generic;

namespace DotNeuralNet
{
  public class BackPropagationTrainer
  {
    private readonly Network _network;
    private readonly double[] _outputGradients;
    private readonly double[] _hiddenGradients;

    public BackPropagationTrainer(Network network)
    {
      _network = network;
      _outputGradients = new double[_network.OutputNodes.Count];
      _hiddenGradients = new double[_network.HiddenNodes.Count];
    }

    public void Train(IEnumerable<BackPropagationTrainingRow> rows, double adjust, int rounds)
    {
      const double errorThreshold = 0.00001;
      foreach (var row in rows)
      {
        for (var r = 0; r < rounds; r++)
        {
          _network.Invalidate();

          for (var i = 0; i < row.Inputs.Length; i++)
          {
            _network.InputNodes[i].Value = row.Inputs[i];
          }
          _network.HiddenNodes[0].Value.ToString();

          if (CalculateErrorForRow(row) < errorThreshold)
          {
            break;
          }
          UpdateWeights(adjust, 0.1, row);
        }
      }
    }

    private void UpdateWeights(double adjust, double momentum, BackPropagationTrainingRow row)
    {
      // Compute output gradients
      for (var i = 0; i < _outputGradients.Length; i++)
      {
        var output = _network.OutputNodes[i].Value;
        var derivative = (1 - output) * output; // derivative of log-sigmoid is y * (1 - y)
        _outputGradients[i] = derivative * (row.Outputs[i] - output); // gradient = (1 - O)(O) * (T-O)
      }

      // Compute hidden gradients
      for (var i = 0; i < _hiddenGradients.Length; i++)
      {
        var hidden = _network.HiddenNodes[i].Value;
        var derivative = (1 - hidden) * (1 + hidden); // derivative of tanh is (1 - y) * (1 + y)
        var sum = .0;
        for (var j = 0; j < _network.OutputNodes.Count; j++)
        {
          // each hidden delta is the sum of numOutput terms
          sum += _outputGradients[j] * _network.OutputNodes[j].Incoming[i].Weight; // each downstream gradient * outgoing weight
        }
        _hiddenGradients[i] = derivative * sum; // hGrad = (1-O)(1+O) * E(oGrads*oWts)
      }

      // Update input to hidden weights
      for (var i = 0; i < _network.InputNodes.Count; ++i)
      {
        for (var j = 0; j < _network.HiddenNodes.Count; ++j)
        {
          var delta = adjust * _hiddenGradients[j] * _network.InputNodes[i].Value; // compute the new delta = "eta * hGrad * input"
          _network.HiddenNodes[j].Incoming[i].Weight += delta; // update
          _network.HiddenNodes[j].Incoming[i].Weight += momentum * _network.HiddenNodes[j].Incoming[i].Delta; // add momentum using previous delta. on first pass old value will be 0.0 but that's OK.
          _network.HiddenNodes[j].Incoming[i].Delta = delta; // save the delta for next time
        }
      }

      // Update hidden biases
      for (var i = 0; i < _network.HiddenNodes.Count; ++i)
      {
        var delta = adjust * _hiddenGradients[i];
        _network.HiddenNodes[i].Bias += delta;
        _network.HiddenNodes[i].Bias += momentum * _network.HiddenNodes[i].BiasDelta;
        _network.HiddenNodes[i].BiasDelta = delta; // save delta
      }

      // Update hidden to output weights
      for (var i = 0; i < _network.HiddenNodes.Count; i++)
      {
        for (var j = 0; j < _network.OutputNodes.Count; ++j)
        {
          var delta = adjust * _outputGradients[j] * _network.HiddenNodes[i].Value;
          _network.OutputNodes[j].Incoming[i].Weight += delta;
          _network.OutputNodes[j].Incoming[i].Weight += momentum * _network.OutputNodes[j].Incoming[i].Delta;
          _network.OutputNodes[j].Incoming[i].Delta = delta;
        }
      }

      // Update hidden to output biases
      for (var i = 0; i < _network.OutputNodes.Count; i++)
      {
        var delta = adjust * _outputGradients[i];
        _network.OutputNodes[i].Bias += delta;
        _network.OutputNodes[i].Bias += momentum * _network.OutputNodes[i].BiasDelta;
        _network.OutputNodes[i].BiasDelta = delta;
      }
    }

    private double CalculateErrorForRow(BackPropagationTrainingRow row)
    {
      var error = .0;
      for (var i = 0; i < row.Outputs.Length; i++)
      {
        var d = row.Outputs[i] - _network.OutputNodes[i].Value;
        error += d * d;
      }
      return Math.Sqrt(error);
    }
  }
}