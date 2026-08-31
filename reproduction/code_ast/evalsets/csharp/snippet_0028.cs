using System;
using IServiceCollectionExtension.LazyLoadService.Test.TestClasses;
using Microsoft.Extensions.DependencyInjection;
using NUnit.Framework;

namespace IServiceCollectionExtension.LazyLoadService.Test
{
    public class Transient_Test
    {
        private IServiceProvider _sp;
        private IServiceCollection _sc;

        [OneTimeSetUp]
        public void OneTimeSetUp()
        {
            _sc = new ServiceCollection();
            _sc.AddTransient<ClassWithLazyDependency>();
            _sc.AddSingleton<StepChecker>();
            _sc.AddLazyScoped<ISomeClass, SomeClass>();
        }

        [SetUp]
        public void SetUp()
        {
            _sp = _sc.BuildServiceProvider();
        }

        [Test]
        public void BuildLazyUserDoesNotInitLazyLoadService()
        {
            // Arrange

            // Act
            var _classWithLazyDependency = _sp.GetRequiredService<ClassWithLazyDependency>();
            var _stepChecker = _sp.GetRequiredService<StepChecker>();

            // Assert
            _stepChecker.PrintPassedSteps();
            Assert.AreEqual(1, _stepChecker.PassedSteps.Length);
            Assert.AreEqual(ClassWithLazyDependency.ConstructorStepName, _stepChecker.PassedSteps[0]);
        }

        [Test]
        public void CallLazyLoadService()
        {
            // Arrange

            // Act
            var _classWithLazyDependency = _sp.GetRequiredService<ClassWithLazyDependency>();
            var _stepChecker = _sp.GetRequiredService<StepChecker>();
            var _assertedResult = _classWithLazyDependency.PingLazyLoadService();

            // Assert
            Assert.AreEqual(SomeClass.PingResponse, _assertedResult);

            _stepChecker.PrintPassedSteps();
            Assert.AreEqual(2, _stepChecker.PassedSteps.Length);
            Assert.AreEqual(ClassWithLazyDependency.ConstructorStepName, _stepChecker.PassedSteps[0]);
            Assert.AreEqual(SomeClass.ConstructorStepName, _stepChecker.PassedSteps[1]);
        }
    }
}