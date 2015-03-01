using System.Runtime.InteropServices;
using DotNeuralNet.Perceptrons;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace DotNeuralNet.Test
{
  [TestClass]
  public class PerceptronTest
  {
    [TestMethod]
    public void TestLogicalAnd()
    {
      var perceptron = new Perceptron(2);
      perceptron.ActivationFunction = x => x >= 1 ? 1 : 0;
      perceptron.Bias = 0;
      perceptron.Weights[0] = 0.5;
      perceptron.Weights[1] = 0.5;

      // 0 & 0 -> 0
      perceptron.Inputs[0] = 0;
      perceptron.Inputs[1] = 0;
      Assert.AreEqual(0, perceptron.Output);
      // 0 & 1 -> 0
      perceptron.Inputs[0] = 0;
      perceptron.Inputs[1] = 1;
      Assert.AreEqual(0, perceptron.Output);
      // 1 & 0 -> 0
      perceptron.Inputs[0] = 1;
      perceptron.Inputs[1] = 0;
      Assert.AreEqual(0, perceptron.Output);
      // 1 & 1 -> 0
      perceptron.Inputs[0] = 1;
      perceptron.Inputs[1] = 1;
      Assert.AreEqual(1, perceptron.Output);
    }

    [TestMethod]
    public void TestLogicalOr()
    {
      var perceptron = new Perceptron(2);
      perceptron.ActivationFunction = x => x >= 1 ? 1 : 0;
      perceptron.Bias = 0;
      perceptron.Weights[0] = 1;
      perceptron.Weights[1] = 1;

      // 0 & 0 -> 0
      perceptron.Inputs[0] = 0;
      perceptron.Inputs[1] = 0;
      Assert.AreEqual(0, perceptron.Output);
      // 0 & 1 -> 0
      perceptron.Inputs[0] = 0;
      perceptron.Inputs[1] = 1;
      Assert.AreEqual(1, perceptron.Output);
      // 1 & 0 -> 0
      perceptron.Inputs[0] = 1;
      perceptron.Inputs[1] = 0;
      Assert.AreEqual(1, perceptron.Output);
      // 1 & 1 -> 0
      perceptron.Inputs[0] = 1;
      perceptron.Inputs[1] = 1;
      Assert.AreEqual(1, perceptron.Output);
    }

    [TestMethod]
    public void TestTrainingBounce()
    {
      var perceptron = new Perceptron(2);
      perceptron.ActivationFunction = x => x >= 1 ? 1 : 0;
      var trainer = new PerceptronTrainer(perceptron);
      var rows = new []
      {
        new PerceptronTrainingRow ( new [] { 0.0, 1.0 }, 0 ),
        new PerceptronTrainingRow ( new [] { 0.5, 1.0 }, 0 ),
        new PerceptronTrainingRow ( new [] { 1.0, 0.0 }, 0 ),
        new PerceptronTrainingRow ( new [] { 1.5, 5.0 }, 0 ),
        new PerceptronTrainingRow ( new [] { 3.0, 3.0 }, 0 ),
        new PerceptronTrainingRow ( new [] { 3.5, 0.0 }, 0 ),
        new PerceptronTrainingRow ( new [] { 1.0, 6.0 }, 1 ),
        new PerceptronTrainingRow ( new [] { 2.0, 9.0 }, 1 ),
        new PerceptronTrainingRow ( new [] { 4.0, 6.0 }, 1 ),
        new PerceptronTrainingRow ( new [] { 5.5, 1.0 }, 1 ),
        new PerceptronTrainingRow ( new [] { 6.0, 4.0 }, 1 ),
        new PerceptronTrainingRow ( new [] { 9.0, 3.0 }, 1 ),
      };
      trainer.Train(rows, 0.01, 500);

      perceptron.Inputs[0] = 2.0;
      perceptron.Inputs[1] = 2.0;
      Assert.AreEqual(0, perceptron.Output);
      perceptron.Inputs[0] = 7.0;
      perceptron.Inputs[1] = 6.0;
      Assert.AreEqual(1, perceptron.Output);
    }
  }
}
