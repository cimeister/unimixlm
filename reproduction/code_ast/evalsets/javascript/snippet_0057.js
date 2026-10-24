/**
 * Created by Shalom on 3/1/2016.
 */
(function () {

    var module = angular.module('bn.project');

    module.controller('CreateProjCtrl', function ($rootScope, $scope, DataFactory, dialogService, Projects) {
        var ctrl = this;
        this.showChildren = function (node, selection) {
            console.log(arguments);
        };

        DataFactory.getProjects().then(function (projects) {
            ctrl.projects = projects;
            ctrl.proj = {
                childs: [],
                factoriesId: [],
                languagesId: [],
                coursesId: [],
                type: "project"
            };
            ctrl.costumer = {};
            ctrl.proj.childs = [];
        });

        $scope.$on('langChange', function (event, newLangs) {
            ctrl.langs = newLangs;
        });

        $scope.$on('costChange', function (event, newCost) {
            ctrl.costumer = DataFactory.getCostumerById(newCost.id)
            ctrl.proj.costumerId = newCost.id;
        });

        $scope.$on('factChange', function (event, newFacts) {
            ctrl.factories = newFacts;
        });

        $scope.$on('userChange', function (event, newUser) {
            ctrl.users = newUser;
        });


        $scope.$on('courseChange', function (event, newCourses) {
            ctrl.proj.coursesId = [];
            console.log('newCourses', newCourses);
            for (var i in newCourses) {
                ctrl.proj.coursesId.push(newCourses[i].id)
            }
            ctrl.courses = newCourses;
            console.log('ctrl.newCourses', ctrl.newCourses);
            ctrl.proj.childs.push(newCourses);
            DataFactory.updateProj(ctrl.proj)
        });


//buttons fnc
        ctrl.showFactories = function ($event) {
            dialogService.showFactories($event)
        };

        ctrl.showCostumers = function ($event) {
            dialogService.showCostumers($event)
        };

        ctrl.showLanguages = function ($event) {
            dialogService.showLanguages($event)
        };

        ctrl.showUsers = function ($event) {
            dialogService.showUsers($event)
        };

        ctrl.showCourses = function ($event) {
            dialogService.showCourses($event)
        };

//create Proj
        ctrl.createProj = function () {
            ctrl.proj.type = 'project';
            if (!Array.isArray(ctrl.langs))ctrl.langs = [];
            ctrl.langs.forEach(function (lang) {
                ctrl.proj.languagesId.push(lang.id)
            });
            if (!Array.isArray(ctrl.factories))ctrl.factories = [];
            ctrl.factories.forEach(function (fact) {
                ctrl.proj.factoriesId.push(fact.id);
            });
            Projects.create(ctrl.proj);
        }
    });
})();