(function() {
  function suggest(urlFunc) {
    var competitionId = $(".schedule-configuration-form").data('competition-id');
    var url = urlFunc(competitionId);

    return function(partial, callback) {
      $.ajax({
        url: url,
        data: {
          q: partial
        },
        success: callback
      });
    }
  }

  var apiUrls = {
    teamSuggestionUrl: function(competitionId) {
      return "/api/v1/competition/" + competitionId + "/team-suggestions";
    },
    judgeSuggestionUrl: function(competitionId) {
      return "/api/v1/competition/" + competitionId + "/judge-suggestions";
    }
  };

  var editorViewModel = {
    'competitionId': $(".schedule-configuration-form").data('competition-id'),
    'matches': ko.observableArray(),
    'allTeams': ko.observableArray()
  };

  editorViewModel.byes = ko.computed(function() {
    var scheduledTeamIds = [];
    editorViewModel.matches().forEach(function(match) {
      if (match.prosecutionTeamId())
        scheduledTeamIds.push(match.prosecutionTeamId());
      if (match.defenseTeamId())
        scheduledTeamIds.push(match.defenseTeamId());
    });

    var byeTeams = editorViewModel.allTeams().filter(function(possibleByeTeam) {
      return ! scheduledTeamIds.includes(possibleByeTeam.id);
    });

    return byeTeams;
  }, editorViewModel);

  editorViewModel.deleteMatch = function(currentItem, event) {
    var index = $(event.target).closest(".match-row").data("index");
    editorViewModel.matches.splice(index, 1);
  }

  window.judicialManager.bindSuggestions = function() {
    function updateMatchIdField(rowElem, fieldName) {
      return function(suggestion) {
        var id = undefined;

        if (suggestion != null) {
          id = suggestion.value;
        }

        var targetIndex = $(rowElem).data('index');
        var targetObservable = editorViewModel.matches()[targetIndex][fieldName];
        targetObservable(id);
      }
    }

    $(".match-row:not(.suggesting)").each(function(index, rowElem) {
      $(rowElem).find(".prosecution-team").elemicaSuggest({
        suggestFunction: suggest(apiUrls.teamSuggestionUrl),
        valueInput: $(rowElem).find(".prosecution-team-id"),
        afterSelect: updateMatchIdField(rowElem, 'prosecutionTeamId')
      });

      $(rowElem).find(".defense-team").elemicaSuggest({
        suggestFunction: suggest(apiUrls.teamSuggestionUrl),
        valueInput: $(rowElem).find(".defense-team-id"),
        afterSelect: updateMatchIdField(rowElem, 'defenseTeamId')
      });

      $(rowElem).find(".presiding-judge").elemicaSuggest({
        suggestFunction: suggest(apiUrls.judgeSuggestionUrl),
        valueInput: $(rowElem).find(".presiding-judge-id"),
        afterSelect: updateMatchIdField(rowElem, 'presidingJudgeId')
      });

      $(rowElem).find(".scoring-judge").elemicaSuggest({
        suggestFunction: suggest(apiUrls.judgeSuggestionUrl),
        valueInput: $(rowElem).find(".scoring-judge-id"),
        afterSelect: updateMatchIdField(rowElem, 'scoringJudgeId')
      });

      $(rowElem).addClass("suggesting");
    });
  }

  judicialManager.editorViewModel = editorViewModel;

  function createMatchObservables(match) {
    var newMatch = {
      prosecutionTeamName: ko.observable((match||{}).prosecutionTeamName),
      prosecutionTeamId: ko.observable((match||{}).prosecutionTeamId),
      defenseTeamName: ko.observable((match||{}).defenseTeamName),
      defenseTeamId: ko.observable((match||{}).defenseTeamId),
      presidingJudgeName: ko.observable((match||{}).presidingJudgeName),
      presidingJudgeId: ko.observable((match||{}).presidingJudgeId),
      scoringJudgeName: ko.observable((match||{}).scoringJudgeName),
      scoringJudgeId: ko.observable((match||{}).scoringJudgeId)
    };

    newMatch.prosecutionTeamData = ko.computed(function() {
      var teamId = newMatch.prosecutionTeamId();
      var teamData = editorViewModel.allTeams().find(function(item) {
        return item.id == teamId;
      });
      return teamData || {id: 0, prosecutionOccurrences: 0, defenseOccurrences: 0};
    });

    newMatch.defenseTeamData = ko.computed(function() {
      var teamId = newMatch.defenseTeamId();
      var teamData = editorViewModel.allTeams().find(function(item) {
        return item.id == teamId;
      });
      return teamData || {id: 0, prosecutionOccurrences: 0, defenseOccurrences: 0};
    });

    editorViewModel.matches.push(newMatch);
  }

  editorViewModel.addMatch = function() {
    createMatchObservables();
    judicialManager.bindSuggestions();
  };

  judicialManager.serializeSchedule = function() {
    return ko.toJSON(editorViewModel.matches);
  }

  judicialManager.setSchedule = function(schedule) {
    editorViewModel.matches.removeAll();

    schedule.forEach(function(match) {
      createMatchObservables(match);
    });

    judicialManager.bindSuggestions();
  }

  judicialManager.setAllTeams = function(allTeams) {
    editorViewModel.allTeams.removeAll();
    allTeams.forEach(function(team) {
      editorViewModel.allTeams.push(team);
    });
  }

  ko.applyBindings(editorViewModel, document.getElementById('schedule-editor-bindings'));
  judicialManager.bindSuggestions();
})();