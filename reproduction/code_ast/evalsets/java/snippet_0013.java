package scrapers.common;

import com.google.common.base.Throwables;
import io.github.bonigarcia.wdm.WebDriverManager;
import org.openqa.selenium.chrome.ChromeDriver;
import org.openqa.selenium.chrome.ChromeOptions;
import scrapers.cityclerk.CityClerkAgent;
import scrapers.core.ScraperAgent;
import scrapers.swedishconsulate.SwedishConsulateAgent;

import java.util.Arrays;


public class Main {
  private static final String SCRAPER_CITY_CLERK_AGENT = "cityclerk";
  private static final String SCRAPER_KONSULATET = "konsulatet";
  private static final String[] ALL_SCRAPERS = {SCRAPER_KONSULATET, SCRAPER_CITY_CLERK_AGENT};

  private static final boolean debug = boolEnv("DEBUG");
  private static final String scraper = strEnv("SCRAPER");

  public static void main(String[] args) {
    ScraperAgent cityClerkAgent = createScraperAgent();
    var notificationAgent = debug ? new LogNotificationAgent() : new SmsNotificationAgent();

    try (var checker = new ScraperLoop(cityClerkAgent, notificationAgent)) {
      checker.run();
    } catch (Throwable t) {
      notificationAgent.sendCrash("CityClerkAgent crashed: " + t);
      Throwables.throwIfUnchecked(t);
      throw new RuntimeException(t);
    }
  }

  static ScraperAgent createScraperAgent() {
    var webDriver = createWebDriver();
    if (scraper.equals(SCRAPER_CITY_CLERK_AGENT)) {
      return new CityClerkAgent(webDriver);
    } else if (scraper.equals(SCRAPER_KONSULATET)) {
      return new SwedishConsulateAgent(webDriver);
    } else {
      throw new RuntimeException("Unknown scraper requested " + scraper + ", set env var SCRAPER to one of " + Arrays.toString(ALL_SCRAPERS));
    }
  }

  static ChromeDriver createWebDriver() {
    WebDriverManager.chromedriver().setup();
    ChromeOptions options = new ChromeOptions();
    options.addArguments("--headless");
    return new ChromeDriver(options);
  }

  public static boolean boolEnv(String name) {
    var value = System.getenv(name);
    if (value == null) {
      return false;
    } else {
      return value.equalsIgnoreCase("true");
    }
  }

  public static String strEnv(String name) {
    var value = System.getenv(name);
    if (value == null) {
      return "";
    } else {
      return value;
    }
  }
}