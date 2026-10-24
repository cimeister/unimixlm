module.export = async(event, context) => {
    try {
        for (const record of event.Records) {
          let messageAttributes = record.messageAttributes;
          console.log('Message Attributtes -->  ', messageAttributes.AttributeNameHere.stringValue);
          console.log('Message Body -->  ', record.body);
          // Do something
        }
      } catch (error) {
        console.log(error);
      }
}