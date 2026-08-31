/*global QUnit, sinon */
sap.ui.define([
	"sap/ui/generic/app/util/ModelUtil",
	"sap/ui/model/odata/v2/ODataModel"
], function(ModelUtil, ODataModel) {
	"use strict";

	QUnit.module("sap.ui.generic.app.util.ModelUtil", {
		beforeEach: function() {
			this.oModel = sinon.createStubInstance(ODataModel);
			this.oUtil = new ModelUtil(this.oModel);
		},
		afterEach: function() {
			this.oUtil.destroy();
		}
	});

	QUnit.test("Shall be instantiable", function(assert) {
		assert.ok(this.oUtil);
	});

	QUnit.test("getEntitySetFromContext throws an exception", function(assert) {
		var bException = false;

		try {
			ModelUtil.getEntitySetFromContext();
		} catch (ex) {
			bException = true;
		}
		assert.ok(bException);
	});

	QUnit.test("getContextFromResponse", function(assert) {
		var oResponse = {};
		this.oModel.getKey = function() {
			return "/Mock(1)";
		};
		this.oModel.getContext = function() {
			return {};
		};

		assert.ok(this.oUtil.getContextFromResponse(oResponse));
	});

	QUnit.test("hasClientMessages returns false", function(assert) {
		assert.ok(!this.oUtil.hasClientMessages());
	});

	QUnit.test("hasClientMessages returns true", function(assert) {
		sap.ui.getCore = function() {
			return {
				getMessageManager: function() {
					return {
						getMessageModel: function() {
							return {
								getData: function() {
									return [ {
										processor: {
											getMetadata: function() {
												return {
													_sClassName: "sap.ui.core.message.ControlMessageProcessor"
												};
											}
										}
									} ];
								}
							};
						}
					};
				}
			};
		};

		assert.ok(this.oUtil.hasClientMessages());
	});

	QUnit.test("Shall be destructible", function(assert) {
		this.oUtil.destroy();
		assert.ok(this.oUtil);
	});
});