///<reference path="../node_modules/@types/mocha/index.d.ts"/>
///<reference path="../node_modules/@types/chai/index.d.ts"/>

import {expect} from "chai";
import {toTranslationTree, toFlatTranslations, ITranslationTree} from "../sources/lib/transformer";

describe("transformer", () => {
    const flatTranslations = {
        "app.title": "Title",
        "app.inner.title": "Inner title",
        "app.inner.button.submit": "Submit",
        "app.inner.button.cancel": "Cancel",
        "dashboard.title": "Dashboard",
        "ok": "Ok",
        "dashboard.list.title": "List"
    };

    describe("toTranslationTree", () => {
        it("should correctly build the tree", () => {
            const tree:any = toTranslationTree(flatTranslations);

            expect(tree).to.deep.equal({
                app: {
                    title: {"en-gb": "Title"},
                    inner: {
                        "title": {"en-gb": "Inner title"},
                        "button": {
                            "submit": {"en-gb": "Submit"},
                            "cancel": {"en-gb": "Cancel"}
                        }
                    }
                },
                ok: {"en-gb": "Ok"},
                dashboard: {
                    title: {"en-gb": "Dashboard"},
                    list: {
                        title: {"en-gb": "List"}
                    }
                }
            });
        });
    });

    describe("toFlatTranslations", () => {
        const flat = {
            "en-gb": {
                "app.title": "Title",
                "app.inner.title": "Inner title",
                "app.inner.button.submit": "Submit",
                "app.inner.button.cancel": "Cancel",
                "dashboard.title": "Dashboard",
                "ok": "Ok",
                "dashboard.list.title": "List"
            }
        };

        const tree: ITranslationTree = {
            "app": {
                "title": {
                    "en-gb": "Title"
                },
                "inner": {
                    "title": {
                        "en-gb": "Inner title"
                    },
                    "button": {
                        "submit": {
                            "en-gb": "Submit"
                        },
                        "cancel": {
                            "en-gb": "Cancel"
                        }
                    }
                }
            },
            "dashboard": {
                "title": {
                    "en-gb": "Dashboard"
                },
                "list": {
                    "title": {
                        "en-gb": "List"
                    }
                }
            },
            "ok": {
                "en-gb": "Ok"
            }
        }


        it("should correctly flatten the tree", () => {
            const flatTranslations = toFlatTranslations(tree);

            expect(flatTranslations).to.deep.equal(flat);
        });
    })
});