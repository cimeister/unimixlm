import { Injectable } from '@angular/core';
import { AngularFirestore } from '@angular/fire/firestore';
import { ActivatedRouteSnapshot, CanActivate } from '@angular/router';

/**
 * Check if request to `.../:periodId/.../:mentorId` is a valid mentor
 * of specific period
 */
@Injectable({ providedIn: 'root' })
export class ValidPeriodOfMentorGuard implements CanActivate {

  constructor(private db: AngularFirestore) { }

  async canActivate({ params }: ActivatedRouteSnapshot) {
    console.log('TODO: need to be implemented through the service');


    const { mentorId, periodId } = params;
    const periodRef = this.db.collection('academic-periods').doc(periodId).ref;

    const snap = await this.db
      .collection('mentors', query => {
        return query
          .where('id', '==', mentorId)
          .where('period.reference', '==', periodRef);
      })
      .get()
      .toPromise();

    // validate if mentor was found or not
    return !snap.empty;
  }
}