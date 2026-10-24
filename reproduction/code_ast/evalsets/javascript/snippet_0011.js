import React from 'react';
import ReactDOM from 'react-dom';
import './index.css';
import * as serviceWorker from './serviceWorker';
import DocsLayout from "@material-docs/core/components/DocsLayout/DocsLayout";
import DocsPages from "@material-docs/core/components/DocsPages/DocsPages";
import DocsMenu from "@material-docs/core/components/DocsMenu/DocsMenu";
import AutoDocsMenu from "@material-docs/core/components/AutoDocsMenu/AutoDocsMenu";
import ReactComponentApiPage from "./lib/components/ReactComponentApiPage/ReactComponentApiPage";
import ReactComponentApiPageProps from "./lib/components/ReactComponentApiPageProps";
import Prop from "./lib/components/Prop/Prop";
import ReactComponentApiPageStyles from "./lib/components/ReactComponentApiPageStyles";
import Style from "./lib/components/Style";
import ReactComponentApiPageImport from "./lib/components/ReactComponentApiPageImport";
import Code from "@material-docs/core/components/Code";
import Markdown from "@material-docs/core/components/Markdown";
import ReactComponentApiPageFooter from "./lib/components/ReactComponentApiPageFooter";
import ReactComponentApiPageSummary from "./lib/components/ReactComponentApiPageSummary";
import H3 from "@material-docs/core/components/H3";

const RU = {
    name: "ru-ru",
    label: "Русский",
    locale: {
        page: {
            text1: "Привет тьма моя давняя подруга",
            text2: "Я пришел снова повидаться с тобой",
        },
        extensions: {

        }
    }
};

const EN = {
    name: "en-us",
    label: "English",
    locale: {
        page: {
            text1: "Hello darkness my old friend",
            text2: "I`ve come to talk with you again",
        },
        extensions: {

        }
    }
};

ReactDOM.render(
    <React.StrictMode>
        <DocsLayout name={"Extension"} noSearchField langs={[EN, RU]} defaultLang={EN}>
            <DocsPages>
                <ReactComponentApiPage name={"Docs Api page"}>
                    <ReactComponentApiPageSummary>
                        <H3>The API documentation of the DemoWithCode React component. Learn more about the props and the CSS customization points.</H3>
                    </ReactComponentApiPageSummary>
                    <ReactComponentApiPageImport>
                        <Code theme={"darcula"}>
                            asdfasfsafs
                        </Code>
                        <Markdown>You can learn more about the difference by [reading this guide]().</Markdown>
                    </ReactComponentApiPageImport>
                    <ReactComponentApiPageProps enableMarkdown>
                        <Prop name={"children"} type={"node"} def={"null"}>Hello __darkness__ my old friend</Prop>
                        <Prop name={"lol"} type={"__node__"} def={"null"}>Hello darkness</Prop>
                        <Prop name={"kek"} type={"node"} def={"null"}>Hello darkness</Prop>
                    </ReactComponentApiPageProps>
                    <ReactComponentApiPageStyles>
                        <Style name={"root"} global={"Mui-root"}>asdfasdf</Style>
                        <Style name={"link"} global={"Mui-link"}>asss</Style>
                    </ReactComponentApiPageStyles>
                    <ReactComponentApiPageFooter>
                        asdfsad
                        <Markdown>asdfasd</Markdown>
                    </ReactComponentApiPageFooter>
                </ReactComponentApiPage>
            </DocsPages>
            <DocsMenu>
                <AutoDocsMenu />
            </DocsMenu>
        </DocsLayout>
    </React.StrictMode>,
    document.getElementById('root')
);

// If you want your app to work offline and load faster, you can change
// unregister() to register() below. Note this comes with some pitfalls.
// Learn more about service workers: https://bit.ly/CRA-PWA
serviceWorker.unregister();