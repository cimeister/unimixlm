import { Component, OnInit, Inject, Input } from '@angular/core';
import {
  OcUserService,
  User,
  ListUser,
  OcUserGroupService,
  ListUserGroupAssignment,
} from '@ordercloud/angular-sdk';
import {
  AppConfig,
  applicationConfiguration,
} from '@app-buyer/config/app.config';
import { Router, ActivatedRoute } from '@angular/router';
import { MeOrderListOptions } from '@app-buyer/order/models/me-order-list-options';
import { OrderStatus } from '@app-buyer/order/models/order-status.model';
import { flatMap, pluck } from 'rxjs/operators';
import { Observable } from 'rxjs';


@Component({
  selector: 'app-user-table',
  templateUrl: './user-table.component.html',
  styleUrls: ['./user-table.component.scss']
})
export class UserTableComponent   implements OnInit {

  users: ListUser;
  selectedUser: User;
  user$:Observable<ListUser>;
  columns = [
    'ID',
    'Username',
    'FirstName',
    'LastName',
    'Email',
    'city',
    'Active',
    'Delete',
    'Assigned',
    'Impersonate',
  ];
  sortBy:string;

  constructor(
    private ocUserService: OcUserService,
    private ocUserGroupService: OcUserGroupService,
    private router: Router,
    private activatedRoute: ActivatedRoute,
    @Inject(applicationConfiguration) private appConfig: AppConfig
  ) {
  
   }

  ngOnInit() {
    
    this.user$=this.listUsers();
  }

  
   listUsers():Observable<any>{
      return this.activatedRoute.queryParamMap.pipe(
        flatMap((queryParamMap)=>{
          this.sortBy = queryParamMap.get('sortBy');
          const listOptions:MeOrderListOptions ={
          sortBy: this.sortBy || undefined,
          search: queryParamMap.get('search') || undefined,
          page: parseInt(queryParamMap.get('page'), 10) || undefined,}
          return this.ocUserService.List("BUYER_ORGANIZATION",listOptions)
        })
      )
  }

  

  
  public sortOrders(sortBy: string): void {
    this.addQueryParam({ sortBy });
  }

  public changePage(page: number): void {
    this.addQueryParam({ page });
  }

  public filterBySearch(search: string): void {
    this.addQueryParam({ search, page: undefined });
  }

  private addQueryParam(newParam: object): void {
    const queryParams = {
      ...this.activatedRoute.snapshot.queryParams,
      ...newParam,
    };
    this.router.navigate([], { queryParams });
  }

}