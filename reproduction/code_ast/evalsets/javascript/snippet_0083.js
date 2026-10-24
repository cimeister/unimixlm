/**
* @jsx React.DOM
*/

'use strict';


/* Controllers */

var dpiControllers = angular.module('dpiControllers', []);

dpiControllers.controller('dpiCtrl', ['$scope',
  function($scope) {
    //$scope.phones = Phone.query();
    $scope.orderProp = 'age';
  }]);