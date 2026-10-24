import { CommonModule } from '@angular/common';
import { NgModule } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ValidationFormModule } from '../validation-form/validation-form.module';
import { CheckboxDefaultComponent } from './checkbox-default/checkbox-default.component';
import { CheckboxFormComponent } from './checkbox-form/checkbox-form.component';

@NgModule({
  declarations: [CheckboxDefaultComponent, CheckboxFormComponent],
  imports: [CommonModule, FormsModule, ValidationFormModule],
  exports: [CheckboxDefaultComponent, CheckboxFormComponent],
})
export class CheckboxModule {}