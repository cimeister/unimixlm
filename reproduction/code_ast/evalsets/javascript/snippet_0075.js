module.exports = function (config) {
  const webpackConfig = require('./webpack.config');
  const webpack = require('webpack');

  webpackConfig.entry = undefined;
  webpackConfig.output = undefined;
  // karma is actually very brittle. The commons chunk plugin as well as the define plugin break it, so we
  // disable these
  webpackConfig.plugins = webpackConfig.plugins
    .filter(function (p) {
      return !(p instanceof webpack.optimize.CommonsChunkPlugin)
    }).filter(function (p) {
      return !(p instanceof webpack.DefinePlugin)
    });
  webpackConfig.devtool = 'inline-source-map';
  config.set({
    basePath: '',
    frameworks: ['mocha', 'chai', 'sinon-chai'],
    plugins: [
      require('karma-mocha'),
      require('karma-chai'),
      require('karma-sinon'),
      require('karma-sinon-chai'),
      require('karma-chrome-launcher'),
      require('karma-phantomjs-launcher'),
      require('karma-mocha-reporter'),
      require('karma-coverage-istanbul-reporter'),
      require('karma-istanbul-threshold'),
      require('karma-webpack'),
      require('karma-sourcemap-loader'),
    ],
    client: {
      clearContext: false // leave Jasmine Spec Runner output visible in browser
    },
    files: [
      {pattern: 'node_modules/sinon/pkg/sinon.js', instrument: false},
      {pattern: 'node_modules/chai/chai.js', instrument: false},
      {pattern: 'node_modules/sinon-chai/lib/sinon-chai.js', instrument: false},
      {pattern: './tests.webpack.js', watched: false}
    ],
    preprocessors: {
      './tests.webpack.js': ['webpack', 'sourcemap']
    },
    mime: {
      'text/x-typescript': ['ts', 'tsx']
    },
    coverageIstanbulReporter: {
      reports: ['html', 'lcovonly', 'json'],
      fixWebpackSourcePaths: true
    },
    istanbulThresholdReporter: {
      src: 'coverage/coverage-final.json',
      reporters: ['text'],
      thresholds: {
        global: {
          statements: 95.01,
          branches: 75.59,
          lines: 91.89,
          functions: 89.23
        },
        each: {
          statements: 75.76,
          branches: 33.33,
          lines: 75,
          functions: 41.67
        }
      }
    },
    angularCli: {
      environment: 'dev'
    },
    reporters: config.angularCli && config.angularCli.codeCoverage
      ? ['progress', 'coverage-istanbul', 'istanbul-threshold']
      : ['progress', 'mocha'],
    port: 9876,
    colors: true,
    logLevel: config.LOG_INFO,
    autoWatch: true,
    browsers: ['Chrome'],
    singleRun: false
  });
};