package kr.lul.util.assertion;

import kr.lul.util.common.MapBuilder;
import org.junit.Test;

import java.util.*;

import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static java.util.Collections.emptySet;
import static org.apache.commons.lang3.RandomStringUtils.random;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

public class AssertsTest {
  @Test
  public void testIsNull() throws Exception {
    Asserts.isNull(null);
    assertThatThrownBy(() -> Asserts.isNull(new Object())).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not null");
  }

  @Test
  public void testIsNotNull() throws Exception {
    assertThatThrownBy(() -> Asserts.isNotNull(null)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is null");
    Asserts.isNotNull(new Object());
  }

  @Test
  public void testZero() throws Exception {
    Asserts.zero(0);
    assertThatThrownBy(() -> Asserts.zero(-1)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not zero")
        .hasMessageContaining("-1");
    assertThatThrownBy(() -> Asserts.zero(1)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not zero")
        .hasMessageContaining("1");

    Asserts.zero(0L);
    assertThatThrownBy(() -> Asserts.zero(-1L)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not zero")
        .hasMessageContaining("-1");
    assertThatThrownBy(() -> Asserts.zero(1L)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not zero")
        .hasMessageContaining("1");
  }

  @Test
  public void testNotZero() throws Exception {
    Asserts.notZero(-1);
    assertThatThrownBy(() -> Asserts.notZero(0)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is zero");
    Asserts.notZero(1);

    Asserts.notZero(-1L);
    assertThatThrownBy(() -> Asserts.notZero(0L)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is zero");
    Asserts.notZero(1L);
  }

  @Test
  public void testPositive() throws Exception {
    assertThatThrownBy(() -> Asserts.positive(-1)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not positive");
    assertThatThrownBy(() -> Asserts.positive(0)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not positive");
    Asserts.positive(1);

    assertThatThrownBy(() -> Asserts.positive(-1L)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not positive");
    assertThatThrownBy(() -> Asserts.positive(0L)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not positive");
    Asserts.positive(1L);

    assertThatThrownBy(() -> Asserts.positive(-1.0)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not positive");
    assertThatThrownBy(() -> Asserts.positive(0.0)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not positive");
    Asserts.positive(1.0);
  }

  @Test
  public void testNotPositive() throws Exception {
    Asserts.notPositive(-1);
    Asserts.notPositive(0);
    assertThatThrownBy(() -> Asserts.notPositive(1)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is positive");

    Asserts.notPositive(-1L);
    Asserts.notPositive(0L);
    assertThatThrownBy(() -> Asserts.notPositive(1L)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is positive");

    Asserts.notPositive(-1.0);
    Asserts.notPositive(0.0);
    assertThatThrownBy(() -> Asserts.notPositive(1.0)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is positive");
  }

  @Test
  public void testNegative() throws Exception {
    Asserts.negative(-1);
    assertThatThrownBy(() -> Asserts.negative(0)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not negative.");
    assertThatThrownBy(() -> Asserts.negative(1)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not negative.");

    Asserts.negative(-1L);
    assertThatThrownBy(() -> Asserts.negative(0L)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not negative.");
    assertThatThrownBy(() -> Asserts.negative(1L)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not negative.");

    Asserts.negative(-1.0);
    assertThatThrownBy(() -> Asserts.negative(0.0)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not negative.");
    assertThatThrownBy(() -> Asserts.negative(1.0)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not negative.");
  }

  @Test
  public void testNotNegative() throws Exception {
    assertThatThrownBy(() -> Asserts.notNegative(-1)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is negative");
    Asserts.notNegative(0);
    Asserts.notNegative(1);

    assertThatThrownBy(() -> Asserts.notNegative(-1L)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is negative");
    Asserts.notNegative(0L);
    Asserts.notNegative(1L);

    assertThatThrownBy(() -> Asserts.notNegative(-1.0)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is negative");
    Asserts.notNegative(0.0);
    Asserts.notNegative(1.0);
  }

  @Test
  public void testEmptyWithCharacterSequence() throws Exception {
    assertThatThrownBy(() -> Asserts.empty((String) null)).isInstanceOf(AssertException.class)
        .hasMessage("string is null.");
    assertThatThrownBy(() -> Asserts.empty((StringBuilder) null)).isInstanceOf(AssertException.class)
        .hasMessage("string is null.");

    Asserts.empty("");
    Asserts.empty(new StringBuilder());

    assertThatThrownBy(() -> Asserts.empty(random(1))).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not empty");
    assertThatThrownBy(() -> Asserts.empty(new StringBuilder(random(1)))).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not empty");
  }

  @Test
  public void testHasLength() throws Exception {
    assertThatThrownBy(() -> Asserts.hasLength(null)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is null");
    assertThatThrownBy(() -> Asserts.hasLength(null)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is null");

    assertThatThrownBy(() -> Asserts.hasLength("")).isInstanceOf(AssertException.class)
        .hasMessageContaining("is empty");
    assertThatThrownBy(() -> Asserts.hasLength(new StringBuilder())).isInstanceOf(AssertException.class)
        .hasMessageContaining("is empty");

    Asserts.hasLength(random(1));
    Asserts.hasLength(new StringBuilder(random(1)));
  }

  @Test
  public void testEmptyWithCollection() throws Exception {
    assertThatThrownBy(() -> Asserts.empty((Collection) null)).isInstanceOf(AssertException.class)
        .hasMessage("collection is null.");

    Asserts.empty(emptyList());
    Asserts.empty(emptySet());

    assertThatThrownBy(() -> Asserts.empty(asList(new Object()))).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not empty");
    assertThatThrownBy(() -> Asserts.empty(new HashSet<>(asList(new Object())))).isInstanceOf(AssertException.class)
        .hasMessageContaining("is not empty");
  }

  @SuppressWarnings("unchecked")
  @Test
  public void testNotEmptyWithCollection() throws Exception {
    assertThatThrownBy(() -> Asserts.notEmpty((Collection) null)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is null");
    assertThatThrownBy(() -> Asserts.notEmpty(emptyList())).isInstanceOf(AssertException.class)
        .hasMessageContaining("is empty");
    assertThatThrownBy(() -> Asserts.notEmpty(emptySet())).isInstanceOf(AssertException.class)
        .hasMessageContaining("is empty");

    Asserts.notEmpty(asList(new Object()));
    Asserts.notEmpty(new HashSet<>(asList(new Object())));
  }

  @Test
  public void testEmptyWithMap() throws Exception {
    assertThatThrownBy(() -> Asserts.empty((Map) null)).isInstanceOf(AssertException.class)
        .hasMessage("map is null.");
    Asserts.empty(new HashMap());
    Asserts.empty(new LinkedHashMap());

    assertThatThrownBy(() -> Asserts.empty(MapBuilder.hashmap(new Object(), null).build()))
        .isInstanceOf(AssertException.class)
        .hasMessageContaining("is not empty");
    assertThatThrownBy(() -> Asserts.empty(MapBuilder.hashmap(new Object(), new Object()).build()))
        .isInstanceOf(AssertException.class)
        .hasMessageContaining("is not empty");
  }

  @Test
  public void testNotEmptyWithMap() throws Exception {
    assertThatThrownBy(() -> Asserts.notEmpty((Map) null)).isInstanceOf(AssertException.class)
        .hasMessageContaining("is null");
    assertThatThrownBy(() -> Asserts.notEmpty(new HashMap())).isInstanceOf(AssertException.class)
        .hasMessageContaining("is empty");

    Asserts.notEmpty(MapBuilder.hashmap(new Object(), null).build());
    Asserts.notEmpty(MapBuilder.hashmap(new Object(), new Object()).build());
  }
}