'use strict';

var escapeHTML = require('../Utils').escapeHTML;

var entryFieldDescription = require('./EntryFieldDescription');

var textBox = function (options, defaultParameters) {

  var resource = defaultParameters,
    label = options.label || resource.id,
    canBeShown = !!options.show && typeof options.show === 'function',
    canBeDisabled = !!options.disabled && typeof options.disabled === 'function',
    description = options.description,
    beforeTextBox = options.beforeTextBox ? options.beforeTextBox : '',
    afterTextBox = options.afterTextBox ? options.afterTextBox : '',
    hidden = options.hidden ? options.hidden : false;

  resource.html =
    '<div style="' + (hidden ? 'display: none;' : '') + '">' +
    beforeTextBox +
    '<label for="camunda-' + escapeHTML(resource.id) + '" ' +
    (canBeShown ? 'data-show="isShown"' : '') +
    (canBeDisabled ? 'data-disable="isDisabled"' : '') +
    '>' + label + '</label>' +
    '<div class="bpp-field-wrapper" ' +
    (canBeShown ? 'data-show="isShown"' : '') +
    (canBeDisabled ? 'data-disable="isDisabled"' : '') +
    '>' +
    '<div style="' +
    (canBeDisabled ? 'pointer-events: none; background-color: #EBEBE4;' : '') + '" contenteditable="true" id="camunda-' + escapeHTML(resource.id) + '" ' +
    'name="' + escapeHTML(options.modelProperty) + '" />' +
    '</div>'
    + afterTextBox +
    '</div>';

  if (options.hasOwnProperty('appliesTo')) {
    resource.appliesTo = options.appliesTo;
  }

  // add description below text box entry field
  if (description) {
    resource.html += entryFieldDescription(description);
  }

  if (canBeShown) {
    resource.isShown = function () {
      return options.show.apply(resource, arguments);
    };
  }

  if (canBeDisabled) {
    resource.isDisabled = function () {
      return options.disabled.apply(resource, arguments);
    };
  }

  resource.cssClasses = ['bpp-textbox'];

  return resource;
};

module.exports = textBox;