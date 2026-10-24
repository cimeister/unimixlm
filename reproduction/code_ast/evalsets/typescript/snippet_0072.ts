import * as React from 'react';
import * as ReactDOM from 'react-dom';
import { override } from '@microsoft/decorators';
import { Log } from '@microsoft/sp-core-library';
import {
  BaseApplicationCustomizer, PlaceholderName
} from '@microsoft/sp-application-base';
import { Dialog } from '@microsoft/sp-dialog';

import * as strings from 'BindingDemoApplicationCustomizerStrings';
import MyComponent from '../component/MyComponent';

/**
 * If your command set uses the ClientSideComponentProperties JSON input,
 * it will be deserialized into the BaseExtension.properties object.
 * You can define an interface to describe it.
 */
export interface IBindingDemoApplicationCustomizerProperties {
  // This is an example; replace with your own property
  testMessage: string;
}

/** A Custom Action which can be run during execution of a Client Side Application */
export default class BindingDemoApplicationCustomizer
  extends BaseApplicationCustomizer<IBindingDemoApplicationCustomizerProperties> {
  private LOG_SOURCE = "BindingDemoApplicationCustomizer.ts";

  @override
  public onInit(): Promise<void> {
    Log.info(this.LOG_SOURCE, `Initialized ${strings.Title}`);

    (window as any).init = (window as any).instance || 0;
    (window as any).init++;
    console.log(`Init call: ${(window as any).init}`);

    (window as any).render = 1;

    // navigatedEvent captures page transition
    this.context.application.navigatedEvent.add(this, this.render);

    // changedEvent doesn't handle page transitions ergo the context not correct so this cannot be used.
    //this.context.placeholderProvider.changedEvent.add(this, this.render);

    return Promise.resolve();
  }

  @override
  public onDispose(): Promise<void> {
    // dispose navigated event
    this.context.application.navigatedEvent.remove(this, this.render);

    // changedEvent doesn't handle page transitions ergo the context not correct so this cannot be used.
    //this.context.placeholderProvider.changedEvent.remove(this, this.render);

    return Promise.resolve();
  }

  private render(): void {
    (window as any).render++;
    console.log(`Render call: ${(window as any).render}`);
    let elementId = "AppCustomDemoContainer";
    if (this.context != undefined) {
      let placeholder = this.context.placeholderProvider.tryCreateContent(PlaceholderName.Top, { onDispose: this.onDispose });
      //let placeholder = this.context.placeholderProvider.tryCreateContent(PlaceholderName.Bottom, { onDispose: this.onDispose });
      if (placeholder != undefined) {
        //Because the placeholder isn't cleaned out we need to validate that we're not going to render more than 1x.
        let container = document.getElementById(elementId);
        if (container == undefined) {
          container = document.createElement("DIV");
          container.setAttribute("id", elementId);
          placeholder.domElement.appendChild(container);
          //should we re-render??
        }
        let element = React.createElement(MyComponent, { context: this.context, init: (window as any).init, render: (window as any).render });
        let elements: any = [];
        elements.push(element);
        ReactDOM.render(elements, container);

      } else {
        console.error(`Bottom Placeholder not available! [${strings.Title} - ${this.LOG_SOURCE}]`);
      }
    } else {
      console.error(`Context is undefined in render! [${strings.Title} - ${this.LOG_SOURCE}]`);
    }
  }

}