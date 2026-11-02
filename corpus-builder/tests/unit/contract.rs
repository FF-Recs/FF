use super::*;

#[test]
fn media_codes_are_exact_and_case_sensitive() {
    for (extension, code) in [
        ("jpg", 1),
        ("png", 2),
        ("gif", 3),
        ("webm", 4),
        ("mp4", 6),
        ("webp", 7),
    ] {
        assert_eq!(extension_code(extension).unwrap(), code);
    }
    for (index, rating) in ["s", "q", "e"].iter().enumerate() {
        assert_eq!(rating_code(rating).unwrap(), index as u8 + 1);
    }
    assert!(extension_code("jpeg").is_err());
    assert!(extension_code("JPG").is_err());
    assert!(extension_code("swf").is_err());
    assert!(rating_code("safe").is_err());
    assert!(rating_code("S").is_err());
}
