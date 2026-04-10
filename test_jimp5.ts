import { Jimp } from "jimp";

async function test() {
    try {
        const img = new Jimp({ width: 100, height: 100, color: 0xFFFFFFFF });
        // @ts-ignore
        const buffer = await img.getBuffer("image/jpeg", { quality: 50 });
        console.log("Success getBuffer with options");
    } catch (e) {
        console.error(e);
    }
}
test();
