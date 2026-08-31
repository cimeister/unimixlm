import {$asArray, $map, $pipe} from 'gs-tools/export/collect';
import {cache} from 'gs-tools/export/data';
import {$keyboard, BaseThemedCtrl, Keyboard, SpecialKeys, _p} from 'mask';
import {$h3, $tbody, attributeIn, element, host, integerParser, multi, PersonaContext, renderCustomElement, renderElement, RenderSpec} from 'persona';
import {combineLatest, Observable, of} from 'rxjs';
import {map} from 'rxjs/operators';

import {TriggerSpec, TriggerType} from '../core/trigger-spec';

import {$helpService, HelpContent} from './help-service';
import template from './help-table.html';


export const $helpTable = {
  tag: 'pb-help-table',
  api: {
    index: attributeIn('index', integerParser()),
  },
};

const $ = {
  host: host($helpTable.api),
  title: element('title', $h3, {}),
  content: element('content', $tbody, {
    rows: multi('#rows'),
  }),
};

@_p.customElement({
  ...$helpTable,
  template,
  dependencies: [Keyboard],
})
export class HelpTable extends BaseThemedCtrl<typeof $> {
  constructor(context: PersonaContext) {
    super(context, $);
  }

  protected get renders(): ReadonlyArray<Observable<unknown>> {
    return [
      this.renderers.title.textContent(this.title$),
      this.renderers.content.rows(this.rows$),
    ];
  }

  @cache()
  private get content$(): Observable<HelpContent|null> {
    return combineLatest([
      $helpService.get(this.vine).contents$,
      this.inputs.host.index,
    ])
        .pipe(
            map(([contents, index]) => {
              if (index === undefined) {
                return null;
              }

              return contents[index] ?? null;
            }),
        );
  }

  @cache()
  private get rows$(): Observable<readonly RenderSpec[]> {
    return this.content$
        .pipe(
            map(content => {
              if (!content) {
                return [];
              }

              const rows$list = $pipe(
                  content.actions,
                  $map(({actionName, trigger}) => {
                    const keyboardEl$ = renderCustomElement({
                      spec: $keyboard,
                      attrs: new Map([['a', of('test')]]),
                      inputs: {text: of(triggerKeySpecToString(trigger))},
                      id: {},
                    });
                    const triggerEl$ = renderElement({
                      tag: 'td',
                      children: [keyboardEl$],
                      id: {},
                    });

                    const actionEl$ = renderElement({
                      tag: 'td',
                      textContent: actionName,
                      id: {},
                    });
                    return renderElement({
                      tag: 'tr',
                      children: [triggerEl$, actionEl$],
                      id: {},
                    });
                  }),
                  $asArray(),
              );

              if (rows$list.length <= 0) {
                return [];
              }

              return rows$list;
            }),
        );
  }

  @cache()
  private get title$(): Observable<string> {
    return this.content$.pipe(map(content => content?.tag?.toLocaleLowerCase() ?? ''));
  }
}


function triggerKeySpecToString(triggerSpec: TriggerSpec|null): string {
  if (!triggerSpec) {
    return 'Disabled';
  }

  const keys: string[] = [];
  if (triggerSpec.alt) {
    keys.push(SpecialKeys.ALT);
  }

  if (triggerSpec.ctrl) {
    keys.push(SpecialKeys.CTRL);
  }

  if (triggerSpec.meta) {
    keys.push(SpecialKeys.META);
  }

  if (triggerSpec.shift) {
    keys.push(SpecialKeys.SHIFT);
  }

  keys.push(triggerTypeToString(triggerSpec.type));
  return keys.join(' ');
}

function triggerTypeToString(triggerType: TriggerType): string {
  if (triggerType === TriggerType.CLICK) {
    return '(click)';
  }

  return triggerType;
}