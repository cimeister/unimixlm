BEGIN_MUTEX("mx1");
print("BEGIN_MUTEX");
var val = get("val1");
if (val == null) {
    val = 0;
}
val = val + 1;
put("val1", val);
print("END_MUTEX");
print("VAL:" + val);
END_MUTEX("mx1");