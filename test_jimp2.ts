import { Jimp } from "jimp";

async function test() {
    try {
        const img = new Jimp({ width: 100, height: 100, color: 0xFFFFFFFF });
        img.resize({ w: 50, h: 50 });
        console.log("Success resize");
    } catch (e) {
        console.error(e);
    }
}
test();
