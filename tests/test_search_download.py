import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from prepare_search_sift import fetch


class Response(io.BytesIO):
    status = 206

    def __init__(self,payload,start,stop,total):
        super().__init__(payload)
        self.headers = {"Content-Range":f"bytes {start}-{stop-1}/{total}"}


class SearchDownloadTests(unittest.TestCase):
    def test_parallel_ranges_are_exact_and_corrupt_cache_is_rejected(self):
        rows = 20000
        total = rows*129*4
        payload = (bytes(range(251))*(total//251+1))[:total]
        expected = hashlib.sha256(payload).hexdigest()

        def open_range(request,timeout):
            start,last = map(int,request.headers["Range"].split("=")[1].split("-"))
            return Response(payload[start:last+1],start,last+1,total)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("prepare_search_sift.urllib.request.urlopen",side_effect=open_range):
                path,info = fetch(root,"test.fvecs",rows,expected)
            self.assertEqual(path.read_bytes(),payload)
            self.assertEqual(info["sha256"],expected)
            self.assertEqual(len(list(root.glob("test.fvecs.range-*"))),3)
            with patch("prepare_search_sift.urllib.request.urlopen",side_effect=AssertionError("Cache must avoid network")):
                fetch(root,"test.fvecs",rows,expected)
                path.write_bytes(b"corrupt")
                with self.assertRaises(ValueError):
                    fetch(root,"test.fvecs",rows,expected)


if __name__ == "__main__":
    unittest.main()
