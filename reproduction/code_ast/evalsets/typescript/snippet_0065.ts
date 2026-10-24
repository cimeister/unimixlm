import { HttpClient } from '@angular/common/http';
import { Component, OnInit } from '@angular/core';
import { Router } from '@angular/router';
import { CookieService } from 'ngx-cookie-service';

@Component({
  selector: 'app-review-template',
  templateUrl: './review-template.component.html',
  styleUrls: ['./review-template.component.css']
})
export class ReviewTemplateComponent implements OnInit {

  inputFileName: any;
  currentUser: any;
  httpString : any;
  loadedHtmlFile: any;

  constructor(private router: Router, private http : HttpClient, private cookieService: CookieService) { }

  ngOnInit(): void
  {
    this.inputFileName = this.cookieService.get('Name')?.replace(/['"]+/g, '');
    this.cookieService.delete('Name');
    this.getHtmlFile();

    this.currentUser = JSON.parse(this.cookieService.get('user' || '{}'))[0];

    if (this.currentUser == undefined || this.currentUser.roleid != 1){
      this.router.navigate(['/no-access']);
    }
  }

  //Gets the HTML file from the templates folder
  getHtmlFile()
  {
      this.http
          .get("../../assets/templates/" + this.inputFileName + ".html",
               { responseType: 'text' })
          .subscribe((data : any) => {
              this.httpString = data;
              this.htmlFromString(this.httpString)
          }
      );
  }

  htmlFromString(htmlString : string)
  {
    this.loadedHtmlFile =  document.createElement('template');
    htmlString = htmlString.trim();
    this.loadedHtmlFile.innerHTML = htmlString;
  }

  //TODO: Add function to change the status of a template to 'Approved' or 'Disapproved'.
  reviewTemplate(reviewStatus: string){
    if (reviewStatus == 'approved'){
      this.router.navigate(['/companyadmin-dashboard'])
    }

    if (reviewStatus == 'disapproved')
    {
      this.router.navigate(['/companyadmin-dashboard'])
    }
  }
  
}