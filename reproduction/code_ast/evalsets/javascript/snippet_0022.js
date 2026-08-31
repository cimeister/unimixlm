"use strict";
import React, { Component } from "react";
import {
    StyleSheet,
    View,
    Text,
    Image,
    TouchableOpacity,
    ScrollView,
    StatusBar,
    ImageBackground
} from "react-native";

import { NavigationActions, StackActions } from "react-navigation";

import ConfigAPI from '../api/ConfigAPI';
import { GetAddressService } from '../api/GetAddressService';
import Utils from "../common/Utils";
import * as Constants from '../common/Constants';
import AvatarView from '../components/AvatarView';
import MenuItem, { MENU_ICON } from '../components/MenuItem';
import * as CommonStyleSheet from '../common/StyleSheet';
import { BaseService } from '../api/BaseService';
import Account from '../models/Account';
import FCM, { FCMEvent, RemoteNotificationResult, WillPresentNotificationResult, NotificationType } from "react-native-fcm";
import { Alert, PushNotificationIOS, NativeModules, NativeEventEmitter, AsyncStorage } from 'react-native';
import RNFetchBlob from 'react-native-fetch-blob';
import PopupMessage from '../components/PopupMessage';
var md5 = require('md5');
import i18n from '../translations/i18n';

export default class Menu extends Component {

    constructor(props) {
        super(props);

        this.state = {
            avatarSource: null,
            userName: "",
            email: "",
            bgxWallet: "0",
            bgtWallet: "0",
            digitalShop: "0",
            isAccountActive: true,
            countTwitter: "0 " + i18n.t('NEWS'),
            menuLogo: null
        };

        this.ethAddress = "";
        this.bgtAddress = "";
        this.scrollView = null;

        this.account = null;
    }

    static navigationOptions = ({ navigation }) => ({
        header: null
    });

    async componentDidMount() {

        var account = await Utils.getDataWithKey(Constants.KEY_USER);
        this.localPassword = await Utils.getDataWithKey(Constants.KEY_PASSWORD);

        let interfaceCode =  await Utils.getDataWithKey(Constants.KEY_DATA_THEME);

        if (interfaceCode!= null && interfaceCode.logo != null) {
            this.setState({
                menuLogo: ConfigAPI.DOMAIN + "/" + interfaceCode.logo
            });
        }

        FCM.on(FCMEvent.Notification, async(notif) => {
            account = await Utils.getDataWithKey(Constants.KEY_USER);
            this.localPassword = await Utils.getDataWithKey(Constants.KEY_PASSWORD);

            if (account == null) {
                return;
            }

            if (notif == null) {
                return;
            }

            let data = notif.data;

            if (data == null) {
                return;
            }

            let dataObject = JSON.parse(data);
            let type = dataObject.type;

            if (type == null) {
                return;
            }

            if (type == 'activation') {
                this.setState({
                    isAccountActive: true
                });

                account.active = "1";
                Utils.saveDataWithKey(Constants.KEY_USER, account);
            } else {
                // alert(dataObject.profile.password + '     ' + md5(this.localPassword.toString()));
                if (dataObject.profile.password != md5(this.localPassword.toString())) {

                    Utils.showAlert(i18n.t('MESSAGE_YOUR_PASSWORD_HAS_CHANGE'), true, {
                        done: () => {
                            this.buttonLogOut_clicked();
                        }
                    });
                    return;
                }

                var avatar = dataObject.profile.avatar;
                if (avatar == null || avatar == "") {
                    avatar = null;
                }

                this.setState({
                    avatarSource: avatar,
                    userName: dataObject.profile.username
                });

                this.ethAddress = dataObject.profile.ethereumAddress;
                this.bgtAddress = dataObject.profile.bgtAddress;

                //get DEC balance
                if (Utils.isValidString(this.ethAddress) && Utils.checkEtherAddressValid(this.ethAddress)) {
                    this.requestGetBGTBalance(true, this.ethAddress);
                }

                //get btBalance
                if (Utils.isValidString(this.bgtAddress)) {
                    this.requestGetBGTBalance(false, this.bgtAddress);
                }

                Utils.saveDataWithKey(Constants.KEY_USER, dataObject.profile);
            }
        });
    }

    async componentWillReceiveProps(nextProps) {
        if (nextProps.navigation.state.isDrawerOpen) {
            this.account = await Utils.getDataWithKey(Constants.KEY_USER);
            let publicPrivateKeys = await Utils.getDataWithKey(Constants.KEY_PUBLIC_PRIVATE);

            if (this.account != null) {
                this.setState({
                    avatarSource: this.account.avatar == "" ? null : this.account.avatar,
                    userName: this.account.username,
                    email: this.account.email,
                    isAccountActive: this.account.active == "1" ? true : false
                });
                this.ethAddress = this.account.ethereumAddress;
                this.bgtAddress = this.account.bgtAddress;

                //get DEC balance
                if (Utils.isValidString(this.ethAddress) && Utils.checkEtherAddressValid(this.ethAddress)) {
                    this.requestGetBGTBalance(true, this.ethAddress);
                }

                //get btBalance
                if (Utils.isValidString(this.bgtAddress)) {
                    this.requestGetBGTBalance(false, this.bgtAddress);
                }

                this.requestGetShoppingCartList(this.account.id);
            }

            // if (publicPrivateKeys != null) {
            //     let publicHashedKey = publicPrivateKeys.publicKeyHashed.trim();
            //     if (publicHashedKey != null && publicHashedKey != "") {
            //         this.requestGetBGTBalance(publicHashedKey);
            //     }
            // }
            this.requestCountTwitter();

        }

        this.scrollView.scrollTo({ y: 0 });
    }

    //Handle Button
    buttonSetting_clicked = () => {
        let routeName = Utils.getCurrentViewName(this.props.navigation);
        if (routeName == 'ProfileView') {
            this.props.navigation.closeDrawer();
        } else {
            this.props.navigation.navigate("ProfileView");
            // this.props.navigation.dispatch(
            //     StackActions.reset({
            //     index: 0,
            //     actions: [NavigationActions.navigate({ routeName: "ProfileView" })]
            //   })
            // );
        }
    }

    buttonBGXWallet_clicked = () => {
        let routeName = Utils.getCurrentViewName(this.props.navigation);
        if (routeName == 'BGXWalletView') {
            this.props.navigation.closeDrawer();
        } else {
            this.props.navigation.navigate("BGXWalletView");
        }
    }

    buttonDigitalSpot_clicked = () => {
        let routeName = Utils.getCurrentViewName(this.props.navigation);
        if (routeName == 'DigitalSpotView') {
            this.props.navigation.closeDrawer();
        } else {
            this.props.navigation.navigate("DigitalSpotView");
        }
    }

    buttonDontMissOut_clicked = () => {
        // this._navigate("DontMissOutView");
        let routeName = Utils.getCurrentViewName(this.props.navigation);
        if (routeName == 'DontMissOutView') {
            this.props.navigation.closeDrawer();
        } else {
            this.props.navigation.navigate("DontMissOutView");
        }
    }

    buttonLogOut_clicked = async() => {
        Utils.saveDataWithKey(Constants.KEY_PASSWORD, null);
        Utils.saveDataWithKey(Constants.KEY_USER, null);
        // Utils.saveDataWithKey(Constants.KEY_PUBLIC_PRIVATE, null);
        const resetAction = StackActions.reset({
            index: 0,
            key: null,
            actions: [NavigationActions.navigate({
                routeName: 'Welcome',
                action: NavigationActions.navigate({ routeName: 'LoginView' })
            })],
        });
        this.props.navigation.dispatch(resetAction);
    }

    buttonShoppingCart_clicked = () => {
        let routeName = Utils.getCurrentViewName(this.props.navigation);
        if (routeName == 'ShoppingCartView') {
            this.props.navigation.closeDrawer();
        } else {
            this.props.navigation.navigate("ShoppingCartView");
        }
    }

    //Request Data
    requestResendEmail = () => {
        let account = new Account(this.account);
        let params = {
            [ConfigAPI.PARAM_METHOD]: ConfigAPI.METHOD_RESEND_ACTIVE_EMAIL,
            [ConfigAPI.PARAM_LOCALE]: i18n.locale,
            [ConfigAPI.PARAM_EMAIL]: account.email,
        };

        let resendActiveEmail = new BaseService();
        resendActiveEmail.setParam(params);
        resendActiveEmail.setCallback(this);
        resendActiveEmail.requestData();
    }

    requestCountTwitter = () => {
        let params = {
            [ConfigAPI.PARAM_METHOD]: ConfigAPI.METHOD_COUNT_TWITTER,
        };

        let countTwitter = new BaseService();
        countTwitter.setParam(params);
        countTwitter.setCallback(this);
        countTwitter.requestData();
    }

    requestGetShoppingCartList = (accountID) => {
        this.setState({ progressView: true });
        let params = {
            [ConfigAPI.PARAM_METHOD]: ConfigAPI.METHOD_GET_LIST_SHOPPING_CART,
            [ConfigAPI.PARAM_LOCALE]: i18n.locale,
            [ConfigAPI.PARAM_USER_ID]: accountID
        };
        let login = new BaseService();
        login.setParam(params);
        login.setCallback(this);
        login.requestData();
    }

    requestGetBGTBalance = (isDEC, address) => {
        
        let params = {
            [ConfigAPI.PARAM_METHOD]: isDEC ? ConfigAPI.METHOD_GET_ADDRESS : ConfigAPI.METHOD_BGT_GET_BALANCE,
            [ConfigAPI.PARAM_API_URL]: ConfigAPI.API_GET_BALANCE_BGT.replace('{hasher_public_key}', address),
        };

        this.setState({ progressView: true });
        let service = new BaseService();
        service.setCallback(this);
        service.setParam(params);
        service.requestDataPostGet(BaseService.METHOD.GET);
    }

    //Request Data Success - Fail
    async onSuccess(code, message, data, method) {
        if (method === ConfigAPI.METHOD_RESEND_ACTIVE_EMAIL) {
            this.popupDialogMessage.showSlideAnimationDialog(i18n.t('MESSAGE_RESEND_EMAIL').replace('{email_address}', this.account.email));
        } else if (method === ConfigAPI.METHOD_GET_LIST_SHOPPING_CART) {
            let count = data.length;
            this.setState({
                digitalShop: count
            });
        } else if (method === ConfigAPI.METHOD_GET_ADDRESS) {
            if (data == "0") {
                this.setState({
                    bgxWallet: 0
                });
            } else {
                this.setState({
                    bgxWallet: Utils.decimalValue(data / 1000000000000000000, 2)
                });
            }
            this.setState({ progressView: false });
        } else if (method === ConfigAPI.METHOD_BGT_GET_BALANCE) {
            let number = data;
            number = Utils.decimalValue(number, 2);
            if (data % 1 == 0) {
                number = parseInt(data, 10);
            }
            this.setState({ bgtWallet: number });
            // this.setState({
            //     bgtWallet: Utils.decimalValue(data, 2)
            // });
        } else if (method === ConfigAPI.METHOD_COUNT_TWITTER) {
            this.setState({
                countTwitter: data + " " + i18n.t('NEWS')
            });
        }
    }

    async onFail(code, message, method) {
    }

    _renderResendActiveEmail = () => {
        if (!this.state.isAccountActive) {
            return (
                <View style={[styles.viewItem, { height: 50, backgroundColor: CommonStyleSheet.THEME_DARCK ? '#121212' : '#f4f4f4' }]}>
                    <View style={[styles.viewItemLogout, { alignItems: 'center', justifyContent: 'center', flexDirection: 'column', backgroundColor: CommonStyleSheet.THEME_DARCK ? '#181818' : 'white' }]}>
                        <Text style={{ color: '#b20000', fontWeight: '600' }}>{i18n.t('YOUR_ACCOUNT_IS_NOT_ACTIVATED')}</Text>
                        <TouchableOpacity onPress={this.requestResendEmail}>
                            <Text style={{ color: '#6666ff', textDecorationLine: 'underline', fontWeight: '600' }}>{i18n.t('RESEND_ANOTHER_VERIFICATION_EMAIL')}</Text>
                        </TouchableOpacity>
                    </View>
                    <View style={[styles.viewLine, { backgroundColor: CommonStyleSheet.THEME_DARCK ? '#131313' : '#e8e8e8' }]} />
                </View>
            );
        }
    }

    _renderViewBGX = () => {
        if (!this.state.isAccountActive) {
            return (
                <View style={[styles.viewItem, { width: '100%', height: 95, position: 'absolute', backgroundColor: CommonStyleSheet.THEME_DARCK ? 'rgba(0, 0, 0, 0.6)' : 'rgba(0, 0, 0, 0.2)' }]} />
            );
        }
    }

    _renderDigitalSpotDeActive = () => {
        if (!this.state.isAccountActive) {
            return (
                <View style={[styles.viewItem, { width: '100%', position: 'absolute', backgroundColor: CommonStyleSheet.THEME_DARCK ? 'rgba(0, 0, 0, 0.6)' : 'rgba(0, 0, 0, 0.2)' }]} />
            );
        }
    }

    render() {
        return (
            <View style={[{ flex: 1 }]}>
                <ImageBackground style={styles.imageBackgroudHeader}
                    source={require("../resource/titleBackground.png")}>
                    <View style={styles.viewAvatar}>
                        <AvatarView containStyle={styles.viewAvatarItem} avatarSource={this.state.avatarSource} backgroundColor={CommonStyleSheet.THEME_DARCK ? 'rgba(0, 0, 0, 0.5)' : 'rgba(244, 244, 244, 0.5)'} isSelectPhoto={false} />
                        <View style={styles.viewText}>
                            <Text style={styles.textUserName} numberOfLines={1}>{this.state.userName}</Text>
                            <Text style={styles.textEmail} numberOfLines={1}>{this.state.email}</Text>
                        </View>
                        <TouchableOpacity style={styles.buttonSetting} onPress={this.buttonSetting_clicked}>
                            <Image style={styles.iconEdit} source={require("../resource/icon_setting.png")} />
                        </TouchableOpacity>
                    </View>

                </ImageBackground>

                {/* View Item */}
                <View style={[styles.viewContentItem, { backgroundColor: CommonStyleSheet.THEME_DARCK ? '#121212' : '#f4f4f4' }]}>
                    <ScrollView ref={ref => this.scrollView = ref}>
                        {/* Account not active */}
                        {this._renderResendActiveEmail()}
                        {/* Location */}

                        {/* BGX WALLET */}
                        <TouchableOpacity disabled={!this.state.isAccountActive} onPress={this.buttonBGXWallet_clicked}>
                            <View style={[styles.viewItem, { height: 95, backgroundColor: CommonStyleSheet.THEME_DARCK ? '#121212' : '#f4f4f4' }]}>
                                <View style={[styles.viewItemWallet, { backgroundColor: CommonStyleSheet.THEME_DARCK ? '#181818' : 'white' }]}>
                                    <Image style={styles.iconImage}
                                        source={CommonStyleSheet.THEME_DARCK ? require("../resource/menuIcon/icon_bgxWallet_dark.png") :
                                            require("../resource/menuIcon/icon_bgxWallet_white.png")} />

                                    <View style={styles.viewTextWallet}>
                                        <Text style={styles.textCurrencyWallet}>{i18n.t('BGX_WALLET')}</Text>
                                        <Text style={[styles.textValueWallet, { color: CommonStyleSheet.THEME_DARCK ? '#ebebeb' : 'black' }]}>{this.state.bgtWallet + ' '}<Text style={[styles.textCurrencyWallet, , { fontSize: 17, color: '#de113e' }]}>{i18n.t('BETA_SYMBOL')}</Text></Text>
                                        <View style={[styles.viewLine, { marginRight: 5, backgroundColor: CommonStyleSheet.THEME_DARCK ? '#2f2f2f' : '#d0d0d0' }]} />
                                        <Text style={[styles.textValueWallet, { color: CommonStyleSheet.THEME_DARCK ? '#ebebeb' : 'black' }]}>{this.state.bgxWallet + ' '}<Text style={styles.textCurrencyWallet}>DEC</Text></Text>
                                    </View>
                                </View>
                            </View>
                            <View style={[styles.viewLine, { backgroundColor: CommonStyleSheet.THEME_DARCK ? '#131313' : '#e8e8e8' }]} />
                            {this._renderViewBGX()}
                        </TouchableOpacity>

                        {/* Digital Shop */}
                        <TouchableOpacity disabled={!this.state.isAccountActive} onPress={this.buttonDigitalSpot_clicked}>
                            <View style={[styles.viewItem, { backgroundColor: CommonStyleSheet.THEME_DARCK ? '#121212' : '#f4f4f4' }]}>
                                <View style={[{ flexDirection: 'row', flex: 1, marginLeft: 10, backgroundColor: CommonStyleSheet.THEME_DARCK ? '#181818' : 'white' }]}>
                                    <Image style={styles.iconImage}
                                        source={CommonStyleSheet.THEME_DARCK ?
                                            require("../resource/menuIcon/icon_digitalShop_dark.png")
                                            : require("../resource/menuIcon/icon_digitalShop_white.png")} />
                                    <View style={styles.viewTextDigitalSpot}>
                                        <Text style={styles.textLabel}>{i18n.t('DIGITAL_SPOT')}</Text>
                                        <TouchableOpacity onPress={this.buttonShoppingCart_clicked}>
                                            <Text style={[styles.textValueDigitalSpot, { color: CommonStyleSheet.THEME_DARCK ? '#ebebeb' : 'black' }]}>
                                                {this.state.digitalShop > 1 ?
                                                    this.state.digitalShop + ' ' + i18n.t('ITEMS_IN_CART')
                                                    : this.state.digitalShop + ' ' + i18n.t('ITEM_IN_CART')
                                                }
                                            </Text>
                                        </TouchableOpacity>
                                    </View>
                                </View>
                            </View>
                            <View style={[styles.viewLine, { backgroundColor: CommonStyleSheet.THEME_DARCK ? '#131313' : '#e8e8e8' }]} />
                            {this._renderDigitalSpotDeActive()}
                        </TouchableOpacity>

                        {/* My Shop */}
                        <MenuItem style={[styles.viewItem, { backgroundColor: CommonStyleSheet.THEME_DARCK ? '#121212' : '#f4f4f4' }]} title={i18n.t('MY_SHOP')} icon={MENU_ICON.MY_SHOP} value={i18n.t('NOT_ACTIVATED')} isActive={false} />

                        {/* Don't miss out */}
                        <MenuItem style={[styles.viewItem, { backgroundColor: CommonStyleSheet.THEME_DARCK ? '#121212' : '#f4f4f4' }]} title={i18n.t('DONT_MISS_OUT')} icon={MENU_ICON.DONT_MISS_OUT} value={this.state.countTwitter} onPress={this.buttonDontMissOut_clicked} />

                        {/* Logout */}
                        <TouchableOpacity onPress={this.buttonLogOut_clicked}>
                            <View style={[styles.viewItem, { backgroundColor: CommonStyleSheet.THEME_DARCK ? '#121212' : '#f4f4f4' }]}>
                                <View style={[styles.viewItemLogout, { backgroundColor: CommonStyleSheet.THEME_DARCK ? '#181818' : 'white' }]}>
                                    <Image style={styles.iconImage}
                                        source={CommonStyleSheet.THEME_DARCK ? require("../resource/menuIcon/icon_logout_dark.png") :
                                            require("../resource/menuIcon/icon_logout_white.png")} />
                                    <Text style={[styles.textValue, { color: CommonStyleSheet.THEME_DARCK ? '#ebebeb' : '#898989', }]}>{i18n.t('LOGOUT')}</Text>
                                </View>
                            </View>
                            <View style={[styles.viewLine, { backgroundColor: CommonStyleSheet.THEME_DARCK ? '#131313' : '#e8e8e8' }]} />
                        </TouchableOpacity>

                    </ScrollView>
                    <View style={styles.viewLogo}>
                    {
                            this.state.menuLogo == null ?
                                <Image source={require('../resource/icon_logo_color_dark.png')} style={styles.image_logo} />
                                : <Image source={{ uri: this.state.menuLogo }} style={styles.image_logo} />
                        }

                        {/* <Text style={styles.textLogo}> LOGO</Text> */}
                    </View>
                </View>

                <PopupMessage
                    ref={(popupDialogMessage) => { this.popupDialogMessage = popupDialogMessage; }}
                    onPress={() => this.popupDialogMessage.dismissSlideAnimationDialog()}
                    typeMenu={true}
                    isConfirm={false}
                    titleButtonRight={i18n.t('OK')}
                    title={'about:blank'}
                />
            </View>
        );
    }
}

const styles = StyleSheet.create({

    imageBackgroudHeader: {
        width: '100%',
        height: 150
    },
    viewAvatar: {
        width: '100%',
        flexDirection: 'row',
        marginTop: 45
    },
    viewAvatarItem: {
        marginLeft: 20,
        width: 90,
        height: 90
    },
    viewText: {
        justifyContent: 'flex-end',
        flex: 1,
        marginLeft: 20,
        marginRight: 20
    },
    textUserName: {
        color: 'white',
        fontSize: 22,
        fontWeight: '300'
    },
    textEmail: {
        color: '#7d7d7d',
        fontSize: 15,
        height: 24,
        marginBottom: -5
    },
    buttonSetting: {
        marginTop: 5,
        right: 10,
        position: 'absolute'
    },
    iconEdit: {
        width: 34,
        height: 34
    },
    viewContentItem: {
        width: '100%',
        flex: 1
    },
    viewItem: {
        height: 80
    },
    iconImage: {
        width: 52,
        height: 52,
        alignSelf: 'center',
        marginLeft: 10
    },
    textValue: {
        fontSize: 20,
        fontWeight: '500',
        alignSelf: 'center',
        marginLeft: 20
    },
    viewTextWallet: {
        alignSelf: 'center',
        marginLeft: 20,
        flex: 1,
        justifyContent: 'space-between'
    },
    viewItemWallet: {
        flexDirection: 'row',
        flex: 1,
        marginLeft: 10
    },
    textValueWallet: {
        fontSize: 18,
        alignSelf: 'flex-start',
        marginTop: 5,
        marginBottom: 5
    },
    textCurrencyWallet: {
        color: '#898989',
        fontSize: 15
    },
    viewLine: {
        height: 1.5
    },
    viewItemLogout: {
        flexDirection: 'row',
        flex: 1,
        marginLeft: 10
    },
    viewLogo: {
        width: '100%',
        height: 60,
        justifyContent: 'center'
    },
    textLogo: {
        alignSelf: 'center',
        fontSize: 28,
        fontWeight: '400',
        color: '#838383',

    },
    viewTextDigitalSpot: {
        height: 45,
        alignSelf: 'center',
        marginLeft: 20,
        flex: 1,
        justifyContent: 'space-between'
    },
    textLabel: {
        fontSize: 15,
        fontWeight: '300',
        color: '#898989'
    },
    textValueDigitalSpot: {
        fontSize: 20,
        fontWeight: '500'
    },
    image_logo: {
        height: 50,
        width: Utils.appSize().width > 400 ? 350 : Utils.appSize().width * 0.9,
        resizeMode: "contain",
    }
});