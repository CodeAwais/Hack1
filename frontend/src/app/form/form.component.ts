import { Component } from '@angular/core';
import {
  FormControl,
  FormGroup,
  ValidationErrors,
  Validators,
  ReactiveFormsModule,
} from '@angular/forms';
import { FormInfo } from './form-info';
import { MatFormFieldModule } from '@angular/material/form-field';

@Component({
  selector: 'app-form',
  imports: [ReactiveFormsModule, MatFormFieldModule],
  templateUrl: './form.component.html',
  styleUrl: './form.component.scss',
})
export class FormComponent {
  form!: FormGroup;
  formInfo: FormInfo = {
    age: 0,
    gender: '',
    symptoms: '',
    image: new File([], ''),
  };
  result: string = '';

  ngOnInit() {
    this.form = new FormGroup({
      age: new FormControl(this.formInfo.age, [
        Validators.required,
        Validators.min(1),
        Validators.max(100),
      ]),
      gender: new FormControl(this.formInfo.gender, Validators.required),
      symptoms: new FormControl(this.formInfo.symptoms, Validators.required),
      image: new FormControl(this.formInfo.image, [Validators.required]),
    });
  }

  imageTypeValidator(allowedTypes: string[]) {
    return (control: FormControl): ValidationErrors | null => {
      const file = control.value as File;
      console.log(file.type);
      if (file && !allowedTypes.includes(file.type)) {
        return {
          invalidFileType: { validTypes: allowedTypes, actualType: file.type },
        };
      }
      return null;
    };
  }

  onSubmit() {
    console.log(this.form.value);
  }
}
