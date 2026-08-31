import { Component, OnInit } from '@angular/core';
import { ModalController } from '@ionic/angular';
import { Venue } from '../../../providers/venue';
import { PageHelper } from '../../../providers/page-helper';
import { Logger } from '../../../providers/logger';
import { Router, ActivatedRoute } from '@angular/router';
import { LTCache } from '../../../providers/lt-cache';

@Component({
  selector: 'app-new-amenity-modal',
  templateUrl: './new-amenity-modal.page.html',
  styleUrls: ['./new-amenity-modal.page.scss'],
})
export class NewAmenityModalPage implements OnInit {
  private _className = 'NewAmenityModalPage';

  // Current signed in users email address
  usersEmail: string;

  // name of the amenity
  amenityName: string = '';

  // name of the selected icon
  selectedIcon = "checkmark-outline";

  //list of possible icons
  iconChoices = ["accessibility"];

  constructor(
    public venueProvider: Venue,
    public cache: LTCache,
    public modalCtrl: ModalController,
    public route: ActivatedRoute,
    public logger: Logger,
    public pageHelper: PageHelper,
    public router: Router
  ) {
    this.usersEmail = cache.getValue(cache.KEY_USER_LOGIN_EMAIL);
    this.usersEmail = this.usersEmail ? this.usersEmail.toLowerCase() : null;
  }

  ngOnInit() { }

  /**
   * This method is called on every page view
   **/
  ionViewWillEnter() {
    this.logger.entry(this._className, 'ionViewWillEnter');

    this.venueProvider.getIcons().subscribe(data => {
      var o = data.icons;
      for (var i = 0; i < o.length; i++) {
        this.iconChoices.push(o[i].name);
      }
    });
  }

  /**
   * Called when a UI selection is made for the icon
   * @param iconName name of the icon
   */
  selectIcon(iconName) {
    this.selectedIcon = iconName;
  }

  /**
   * Close and do not save
   */
  closeModal() {
    this.modalCtrl.dismiss({});
  }

  /**
   * Closes the modal and saves changes
   */
  closeAndSaveSelections() {
    this.logger.entry(this._className, 'closeAndSaveSelections,');

    let amenityDoc = {
      "venueId": null,
      "icon": this.selectedIcon,
      "name": this.amenityName,
      "viewable": true,
      "description": [{
        "lang": "en",
        "name": this.amenityName,
        "description": null
      }]
    };
    this.createAmenity(amenityDoc);
  }

  /**
  * Creates a amenity element
  */
  async createAmenity(amenityDoc) {
    this.logger.entry(this._className, 'createAmenity');
    await this.pageHelper.showSaver();
    await this.venueProvider.createAmenity(amenityDoc).subscribe(
      (resp) => {
        this.pageHelper.hideLoader();
        this.modalCtrl.dismiss({ createdElement: resp });
      },
      (err) => {
        this.pageHelper.processRequestError(err).subscribe((resp) => { });
      }
    );
  }


}