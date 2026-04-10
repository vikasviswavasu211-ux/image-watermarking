import { Jimp } from "jimp";
import fs from "fs";

async function test() {
    const buffer = fs.readFileSync("package.json"); // just a dummy buffer
    try {
        const img = await Jimp.read(buffer);
        console.log("Success");
    } catch (e) {
        console.error(e);
    }
}
test();
