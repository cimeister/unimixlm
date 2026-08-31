import { Component, Input } from '@angular/core';
import { Application } from '../../application.model/application.model';

/**
 * Credits extension request.
 */
@Component({
	selector: 'app-credits-extension-detail',
	templateUrl: './credits-extension-detail.component.html',
})
export class CreditsExtensionDetailComponent {
  @Input() application: Application;
  @Input() is_vo_admin: boolean;
}