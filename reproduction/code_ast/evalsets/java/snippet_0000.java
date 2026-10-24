package leetcode;

import java.util.HashMap;
import java.util.Map;

/**
 * Given a string, find the length of the longest substring without repeating characters.
 * <p>
 * Examples:
 * <p>
 * Given "abcabcbb", the answer is "abc", which the length is 3.
 * <p>
 * Given "bbbbb", the answer is "b", with the length of 1.
 * <p>
 * Given "pwwkew", the answer is "wke", with the length of 3. Note that the answer must be a substring, "pwke" is a
 * subsequence and not a substring.
 */
public class LongestSubstring {

    public int logestSubstring(String text) {

        int longest = 0;
        Map<Character, Integer> counterMap = new HashMap<>();

        int i = 0;
        int j = 0;

        while (i < text.length() && j < text.length()) {
            char currentChar = text.charAt(j);

            counterMap.compute(currentChar, (ch, counter) -> counter == null ? 1 : ++counter);

            if (counterMap.get( currentChar) == 1) {
                if (longest < (j - i + 1)) {
                    longest = (j - i + 1);
                }
            } else {
                while (i <= j) {

                    if (counterMap.get(text.charAt(i)) > 1) {
                        counterMap.compute(text.charAt(i), (ch, counter) -> --counter);
                        i++;
                        break;
                    }
                    counterMap.compute(text.charAt(i), (ch, counter) -> --counter);
                    i++;
                }
            }

            j++;
        }


        return longest;
    }
}