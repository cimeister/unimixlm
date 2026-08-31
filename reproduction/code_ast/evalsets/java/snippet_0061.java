package uk.gov.service.bluebadge.test.acceptance.steps.site;

import cucumber.api.java.en.And;
import java.util.ArrayList;
import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import uk.gov.service.bluebadge.test.acceptance.pages.site.CommonPage;
import uk.gov.service.bluebadge.test.acceptance.pages.site.WhereCanWalkPage;
import uk.gov.service.bluebadge.test.acceptance.steps.AbstractSpringSteps;
import uk.gov.service.bluebadge.test.acceptance.steps.CommonSteps;

public class WhereCanWalkSteps extends AbstractSpringSteps {

  private CommonSteps commonSteps;
  private CommonPage commonPage;

  @Autowired
  public WhereCanWalkSteps(CommonPage commonPage, CommonSteps commonSteps) {
    this.commonPage = commonPage;
    this.commonSteps = commonSteps;
  }

  @And("^I validate where can you walk page for a \"(yourself|someone else)\" application$")
  public void iValidateWhereCanYouWalkPage(String applicant) {
    verifyPageContent(applicant);
    //To validate when no data in entered and Continue button is clicked
    List<String> messages = new ArrayList<>();
    messages.add(WhereCanWalkPage.VALIDATION_MESSAGE_FOR_EMPTY_PLACE);
    messages.add(WhereCanWalkPage.VALIDATION_MESSAGE_FOR_EMPTY_TIME);
    commonSteps.iVerifyMultipleValidationMessages(messages);

    //To validate am invalid place
    commonPage.findPageElementById(WhereCanWalkPage.WHERE_PLACE).click();
    commonPage
        .findPageElementById(WhereCanWalkPage.WHERE_PLACE)
        .sendKeys(WhereCanWalkPage.INVALID_PLACE);
    commonPage.findPageElementById(WhereCanWalkPage.WHERE_TIME).click();
    commonPage
        .findPageElementById(WhereCanWalkPage.WHERE_TIME)
        .sendKeys(WhereCanWalkPage.VALID_TIME);
    commonSteps.iVerifyValidationMessage(WhereCanWalkPage.VALIDATION_MESSAGE_FOR_INVALID_PLACE);

    //To validate am invalid time
    commonPage.findPageElementById(WhereCanWalkPage.WHERE_PLACE).click();
    commonPage.findPageElementById(WhereCanWalkPage.WHERE_PLACE).clear();
    commonPage
        .findPageElementById(WhereCanWalkPage.WHERE_PLACE)
        .sendKeys(WhereCanWalkPage.VALID_PLACE);
    commonPage.findPageElementById(WhereCanWalkPage.WHERE_TIME).click();
    commonPage.findPageElementById(WhereCanWalkPage.WHERE_TIME).clear();
    commonPage
        .findPageElementById(WhereCanWalkPage.WHERE_TIME)
        .sendKeys(WhereCanWalkPage.INVALID_TIME);
    commonSteps.iVerifyValidationMessage(WhereCanWalkPage.VALIDATION_MESSAGE_FOR_INVALID_TIME);

    //Enter valid values and move on
    commonPage.findPageElementById(WhereCanWalkPage.WHERE_TIME).click();
    commonPage.findPageElementById(WhereCanWalkPage.WHERE_TIME).clear();
    commonPage
        .findPageElementById(WhereCanWalkPage.WHERE_TIME)
        .sendKeys(WhereCanWalkPage.VALID_TIME);
    commonSteps.iClickOnContinueButton();
  }

  public void verifyPageContent(String applicant) {

    commonSteps.iShouldSeeTheCorrectURL(WhereCanWalkPage.PAGE_URL);

    if ("yourself".equals(applicant.toLowerCase())) {
      commonSteps.thenIShouldSeePageTitledWithGovUkSuffix(WhereCanWalkPage.PAGE_TITLE);
      commonSteps.iShouldSeeTheHeading(WhereCanWalkPage.PAGE_HEADING);
    } else {
      commonSteps.thenIShouldSeePageTitledWithGovUkSuffix(WhereCanWalkPage.PAGE_TITLE_SOMEONE_ELSE);
      commonSteps.iShouldSeeTheHeading(WhereCanWalkPage.PAGE_HEADING_SOMEONE_ELSE);
    }
  }
}